#!/usr/bin/env python3
"""Train a lightweight model to refine motion vectors using optical flow teachers."""

import argparse
import glob
import json
import pickle
import random
import sys
import time
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import h5py

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import Mamba components
sys.path.insert(0, str(REPO_ROOT / "model"))
try:
    from multimodal_resampler.mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    Mamba = None
    print("Warning: Mamba module not available. Use --model-type cnn to use CNN-based refiner.")


@dataclass
class MotionVectorSample:
    motion_vectors: np.ndarray  # (H, W, 2)
    teacher_flow: np.ndarray  # (H, W, 2)
    visibility: np.ndarray  # (H, W, 1)
    video_id: str
    frame_idx: int
    source_file: Path


class MotionVectorRefineDataset(Dataset):
    """Dataset pairing motion vectors with optical-flow teacher targets."""

    def __init__(
        self,
        pkl_paths: List[str],
        limit: Optional[int] = None,
        use_visibility: bool = True,
        verbose: bool = False,
        chunk_fraction: float = 1.0,
    ) -> None:
        self.use_visibility = use_visibility
        self.verbose = verbose
        self.samples: List[MotionVectorSample] = []

        expanded_paths = []
        for path in pkl_paths:
            if any(ch in path for ch in "*?"):
                path_obj = Path(path)
                if path_obj.is_absolute():
                    root = path_obj.anchor
                    pattern = path_obj.relative_to(path_obj.anchor)
                    expanded = list(Path(root).glob(str(pattern)))
                else:
                    expanded = list(Path().glob(path))
            else:
                expanded = [Path(path)]
            expanded_paths.extend(expanded)

        for pkl_path in expanded_paths:
            if not pkl_path.exists():
                if self.verbose:
                    print(f"[dataset] Missing file: {pkl_path}")
                continue
            if self.verbose:
                print(f"[dataset] Loading {pkl_path}")
            with pkl_path.open("rb") as f:
                payload = pickle.load(f)

            num_frames = 0
            for video_id, record in payload.items():
                pb_motion_vectors = record.get("pb_motion_vectors")
                if pb_motion_vectors is None:
                    continue
                pb_indices = record.get("pb_frame_indices", [])
                if len(pb_indices) != len(pb_motion_vectors):
                    continue

                flow_lookup = {}
                for gop in record.get("gop_flows", []):
                    for pb_flow in gop.get("pb_flows", []):
                        frame_index = pb_flow["frame_idx"]
                        flow_lookup[frame_index] = np.asarray(pb_flow["flow"], dtype=np.float32)

                for local_idx, frame_idx in enumerate(pb_indices):
                    if frame_idx not in flow_lookup:
                        continue
                    mv = np.asarray(pb_motion_vectors[local_idx], dtype=np.float32)
                    teacher = flow_lookup[frame_idx]
                    grid_h, grid_w = mv.shape[:2]
                    try:
                        teacher = teacher.reshape(grid_h, grid_w, -1)
                    except ValueError:
                        continue

                    teacher_flow = teacher[..., :2]
                    visibility = teacher[..., 2:3]

                    self.samples.append(
                        MotionVectorSample(
                            motion_vectors=mv,
                            teacher_flow=teacher_flow,
                            visibility=visibility,
                            video_id=video_id,
                            frame_idx=int(frame_idx),
                            source_file=pkl_path,
                        )
                    )
                    num_frames += 1

                    if limit is not None and len(self.samples) >= limit:
                        if self.verbose:
                            print(f"[dataset] Reached limit {limit}, stopping load")
                        return

            if self.verbose:
                print(f"[dataset] Added {num_frames} frames from {pkl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        mv = torch.from_numpy(sample.motion_vectors).permute(2, 0, 1).contiguous()
        teacher_flow = torch.from_numpy(sample.teacher_flow).permute(2, 0, 1).contiguous()
        visibility = torch.from_numpy(sample.visibility).permute(2, 0, 1).contiguous()

        output = {
            "motion_vectors": mv,
            "teacher_flow": teacher_flow,
            "visibility": visibility,
            "video_id": sample.video_id,
            "frame_idx": sample.frame_idx,
            "source_file": str(sample.source_file),
        }

        if not self.use_visibility:
            output.pop("visibility")

        return output


class MotionFlowAlignmentDataset(Dataset):
    """Dataset aligning raw motion vectors from HDF5 chunks with CoTracker3 optical flow."""

    def __init__(
        self,
        mv_root: str,
        optical_flow_root: str,
        limit: Optional[int] = None,
        use_visibility: bool = True,
        verbose: bool = False,
        chunk_fraction: float = 1.0,
    ) -> None:
        self.use_visibility = use_visibility
        self.verbose = verbose
        self.samples: List[MotionVectorSample] = []
        self.stats: Dict[str, int] = {
            "chunks_scanned": 0,
            "samples_loaded": 0,
            "missing_flow": 0,
            "missing_video": 0,
            "shape_mismatch": 0,
        }

        if not (0.0 < chunk_fraction <= 1.0):
            raise ValueError(f"chunk_fraction must be in (0, 1], got {chunk_fraction}")
        self.chunk_fraction = float(chunk_fraction)

        mv_root_path = Path(mv_root).expanduser()
        flow_root_path = Path(optical_flow_root).expanduser()
        if not mv_root_path.exists():
            raise FileNotFoundError(f"Motion vector root not found: {mv_root_path}")
        if not flow_root_path.exists():
            raise FileNotFoundError(f"Optical flow root not found: {flow_root_path}")

        dataset_index, chunk_index = self._build_flow_index(flow_root_path)
        matched_chunks = 0

        h5_paths = sorted(mv_root_path.rglob('*.h5'))
        total_chunks = len(h5_paths)
        if total_chunks == 0:
            print(f"[dataset] No motion vector chunks found under {mv_root_path}")
            return

        print(f"[dataset] Found {total_chunks} motion vector chunks under {mv_root_path}")

        if self.chunk_fraction < 1.0:
            target_chunks = max(1, math.ceil(total_chunks * self.chunk_fraction))
            if target_chunks < total_chunks:
                if self.verbose:
                    print(
                        f"[dataset] Using chunk_fraction={self.chunk_fraction:.3f}; sampling first {target_chunks} chunks"
                    )
                h5_paths = h5_paths[:target_chunks]
                total_chunks = len(h5_paths)

        scan_start = time.perf_counter()

        for idx, h5_path in enumerate(h5_paths, start=1):
            dataset_key = self._chunk_key(mv_root_path, h5_path)
            flow_path = self._match_flow_path(dataset_index, chunk_index, dataset_key)
            if self.verbose or idx == 1 or idx == total_chunks or idx % max(1, total_chunks // 10 or 1) == 0:
                print(
                    f"[dataset] [{idx}/{total_chunks}] processing {h5_path.relative_to(mv_root_path)} | "
                    f"matched_chunks={matched_chunks} samples={len(self.samples)}"
                )

            if flow_path is None or not flow_path.exists():
                self.stats["missing_flow"] += 1
                if self.verbose:
                    print(f"[dataset] Flow file missing for chunk {dataset_key[0]}/{dataset_key[1]}")
                continue

            added = self._load_chunk(h5_path, flow_path, limit)
            if added > 0:
                matched_chunks += 1
                self.stats["samples_loaded"] += added
            self.stats["chunks_scanned"] += 1

            if limit is not None and len(self.samples) >= limit:
                break

        if self.verbose:
            print(f"[dataset] Loaded {len(self.samples)} samples from {matched_chunks} matched chunks")
            print(f"[dataset] Stats: {self.stats}")
        else:
            scan_duration = time.perf_counter() - scan_start
            print(
                f"[dataset] Completed scan in {scan_duration:.1f}s | "
                f"matched_chunks={matched_chunks} samples={len(self.samples)} missing_flow={self.stats['missing_flow']}"
            )

        random.shuffle(self.samples)

    @staticmethod
    def _chunk_key(root: Path, h5_path: Path) -> Tuple[str, str]:
        rel = h5_path.relative_to(root)
        parts = rel.parts
        dataset_name = parts[0] if parts else ''
        return dataset_name, h5_path.stem

    @staticmethod
    def _build_flow_index(flow_root: Path) -> Tuple[Dict[Tuple[str, str], Path], Dict[str, List[Path]]]:
        dataset_index: Dict[Tuple[str, str], Path] = {}
        chunk_index: Dict[str, List[Path]] = defaultdict(list)
        for pkl_path in flow_root.rglob('*.pkl'):
            if pkl_path.name.endswith('_summary.json'):
                continue
            rel = pkl_path.relative_to(flow_root)
            parts = rel.parts
            dataset_name = parts[0] if parts else ''
            key = (dataset_name, pkl_path.stem)
            dataset_index.setdefault(key, pkl_path)
            chunk_index[pkl_path.stem].append(pkl_path)
        return dataset_index, chunk_index

    @staticmethod
    def _match_flow_path(
        dataset_index: Dict[Tuple[str, str], Path],
        chunk_index: Dict[str, List[Path]],
        key: Tuple[str, str],
    ) -> Optional[Path]:
        if key in dataset_index:
            return dataset_index[key]
        candidates = chunk_index.get(key[1], [])
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _load_chunk(self, h5_path: Path, flow_path: Path, limit: Optional[int]) -> int:
        added = 0
        with h5py.File(h5_path, 'r') as h5_file:
            data_group = h5_file.get('data')
            if data_group is None:
                return 0

            with flow_path.open('rb') as f:
                payload = pickle.load(f)

            for video_id, record in payload.items():
                if video_id not in data_group:
                    self.stats["missing_video"] += 1
                    continue

                mv_dataset = data_group[video_id]
                grid_h, grid_w = mv_dataset.shape[1:3]

                flow_lookup: Dict[int, np.ndarray] = {}
                for gop in record.get('gop_flows', []):
                    for pb_flow in gop.get('pb_flows', []):
                        frame_index = int(pb_flow['frame_idx'])
                        flow_arr = np.asarray(pb_flow['flow'], dtype=np.float32)
                        try:
                            flow_arr = flow_arr.reshape(grid_h, grid_w, -1)
                        except ValueError:
                            self.stats["shape_mismatch"] += 1
                            continue
                        flow_lookup[frame_index] = flow_arr

                pb_indices = record.get('pb_frame_indices', [])
                for frame_idx in pb_indices:
                    if limit is not None and len(self.samples) >= limit:
                        return added
                    frame_idx = int(frame_idx)
                    if frame_idx >= mv_dataset.shape[0]:
                        continue
                    flow_arr = flow_lookup.get(frame_idx)
                    if flow_arr is None:
                        continue
                    mv = np.asarray(mv_dataset[frame_idx], dtype=np.float32)
                    if mv.shape[:2] != flow_arr.shape[:2]:
                        self.stats["shape_mismatch"] += 1
                        continue

                    teacher_flow = flow_arr[..., :2]
                    if self.use_visibility and flow_arr.shape[-1] >= 3:
                        visibility = flow_arr[..., 2:3]
                    else:
                        visibility = np.ones((*teacher_flow.shape[:2], 1), dtype=np.float32)

                    self.samples.append(
                        MotionVectorSample(
                            motion_vectors=mv,
                            teacher_flow=teacher_flow,
                            visibility=visibility,
                            video_id=video_id,
                            frame_idx=frame_idx,
                            source_file=flow_path,
                        )
                    )
                    added += 1
                    if limit is not None and len(self.samples) >= limit:
                        return added

        return added


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        mv = torch.from_numpy(sample.motion_vectors).permute(2, 0, 1).contiguous()
        teacher_flow = torch.from_numpy(sample.teacher_flow).permute(2, 0, 1).contiguous()
        visibility = torch.from_numpy(sample.visibility).permute(2, 0, 1).contiguous()
        if not self.use_visibility:
            visibility = torch.ones_like(teacher_flow[:1])

        return {
            "motion_vectors": mv,
            "teacher_flow": teacher_flow,
            "visibility": visibility,
            "video_id": sample.video_id,
            "frame_idx": sample.frame_idx,
            "source_file": str(sample.source_file),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return self.act(out + x)


class MotionVectorRefiner(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.head = nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)

    def forward(self, motion_vectors: torch.Tensor) -> torch.Tensor:
        features = self.stem(motion_vectors)
        features = self.blocks(features)
        return self.head(features)


class MambaMotionVectorRefiner(nn.Module):
    """Motion vector refiner using bidirectional Mamba block."""

    def __init__(self, hidden_dim: int = 64, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()

        if Mamba is None:
            raise RuntimeError("Mamba module not available. Install required dependencies.")

        # Input projection: 2 channels (mv_x, mv_y) to hidden_dim
        self.stem = nn.Linear(2, hidden_dim)

        # Single bidirectional Mamba block
        self.mamba = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            bimamba=True,  # Enable bidirectional processing
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection: hidden_dim back to 2 channels
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, motion_vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_vectors: (B, 2, H, W) tensor of motion vectors
        Returns:
            refined_vectors: (B, 2, H, W) tensor of refined motion vectors
        """
        B, C, H, W = motion_vectors.shape

        # Reshape to (B, H*W, 2) for sequence processing
        x = motion_vectors.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Project to hidden dimension
        x = self.stem(x)

        # Apply bidirectional Mamba
        residual = x
        x = self.mamba(x)
        x = residual + x  # Residual connection

        # Normalize
        x = self.norm(x)

        # Project back to 2 channels
        x = self.head(x)

        # Reshape back to (B, 2, H, W)
        x = x.reshape(B, H, W, 2).permute(0, 3, 1, 2)

        return x


class TransformerMotionVectorRefiner(nn.Module):
    """Lightweight transformer refiner that models spatial relationships within motion grids."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stem = nn.Conv2d(2, hidden_dim, kernel_size=1)
        feedforward_dim = max(int(hidden_dim * ffn_multiplier), hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Conv2d(hidden_dim, 2, kernel_size=1)

    def forward(self, motion_vectors: torch.Tensor) -> torch.Tensor:
        B, _, H, W = motion_vectors.shape
        x = self.stem(motion_vectors)  # [B, hidden_dim, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, HW, hidden_dim]
        x = self.encoder(x)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.hidden_dim, H, W)
        return self.head(x)


def masked_l2(pred: torch.Tensor, target: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
    """Compute masked L2 loss for motion vectors."""
    weight = visibility
    diff = (pred - target) ** 2
    weighted_diff = diff * weight
    denom = weight.sum() * target.shape[1] + 1e-6
    return torch.sqrt(weighted_diff.sum() / denom + 1e-9)


def masked_epe(pred: torch.Tensor, target: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
    squared = (pred - target) ** 2
    epe = torch.sqrt(squared.sum(dim=1, keepdim=True) + 1e-9)
    weighted = epe * visibility
    denom = visibility.sum() + 1e-6
    return weighted.sum() / denom


def collate_fn(batch):
    motion_vectors = torch.stack([item["motion_vectors"] for item in batch], dim=0)
    teacher_flow = torch.stack([item["teacher_flow"] for item in batch], dim=0)
    if "visibility" in batch[0]:
        visibility = torch.stack([item["visibility"] for item in batch], dim=0)
    else:
        visibility = torch.ones(
            motion_vectors.size(0),
            1,
            motion_vectors.size(2),
            motion_vectors.size(3),
            dtype=motion_vectors.dtype,
            device=motion_vectors.device,
        )
    meta = {
        "video_id": [item["video_id"] for item in batch],
        "frame_idx": [item["frame_idx"] for item in batch],
        "source_file": [item["source_file"] for item in batch],
    }
    return {
        "motion_vectors": motion_vectors,
        "teacher_flow": teacher_flow,
        "visibility": visibility,
        "meta": meta,
    }


def create_dataloaders(args):
    load_start = time.perf_counter()
    if args.pkl_paths:
        dataset = MotionVectorRefineDataset(
            args.pkl_paths,
            limit=args.max_samples,
            use_visibility=not args.disable_visibility,
            verbose=args.log_dataset,
        )
    else:
        dataset = MotionFlowAlignmentDataset(
            mv_root=args.mv_root,
            optical_flow_root=args.optical_flow_root,
            limit=args.max_samples,
            use_visibility=not args.disable_visibility,
            verbose=args.log_dataset,
            chunk_fraction=args.dataset_fraction,
        )

    load_duration = time.perf_counter() - load_start
    print(f"[setup] Loaded dataset with {len(dataset)} samples in {load_duration:.1f}s")
    if len(dataset) == 0:
        raise RuntimeError("No samples found. Check dataset arguments.")

    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)

    val_size = int(len(indices) * args.val_split)
    val_size = max(val_size, 1)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    if len(train_indices) == 0:
        raise RuntimeError("Not enough samples for training after split.")

    if args.train_limit:
        train_indices = train_indices[: args.train_limit]
    if args.val_limit:
        val_indices = val_indices[: args.val_limit]

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    print(
        "[setup] Split samples -> train: {} | val: {}".format(
            len(train_ds), len(val_ds)
        )
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, device, args):
    model.train()
    total_loss = 0.0
    total_epe = 0.0
    total_batches = 0

    for batch in tqdm(loader, desc="train", leave=False):
        mv = batch["motion_vectors"].to(device)
        teacher_flow = batch["teacher_flow"].to(device)
        visibility = batch["visibility"].to(device)

        optimizer.zero_grad()

        prediction = model(mv)
        if args.predict_residual:
            prediction = prediction + mv

        loss = masked_l2(prediction, teacher_flow, visibility)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            epe = masked_epe(prediction, teacher_flow, visibility)

        total_loss += loss.item()
        total_epe += epe.item()
        total_batches += 1

    return total_loss / total_batches, total_epe / total_batches


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    total_loss = 0.0
    total_epe = 0.0
    total_batches = 0

    for batch in tqdm(loader, desc="val", leave=False):
        mv = batch["motion_vectors"].to(device)
        teacher_flow = batch["teacher_flow"].to(device)
        visibility = batch["visibility"].to(device)

        prediction = model(mv)
        if args.predict_residual:
            prediction = prediction + mv

        loss = masked_l2(prediction, teacher_flow, visibility)
        epe = masked_epe(prediction, teacher_flow, visibility)

        total_loss += loss.item()
        total_epe += epe.item()
        total_batches += 1

    return total_loss / total_batches, total_epe / total_batches


def main():
    parser = argparse.ArgumentParser(description="Motion vector refinement training")
    parser.add_argument("--pkl-paths", nargs="+", help="Glob(s) or files with teacher optical flow", required=False)
    parser.add_argument("--mv-root", type=str, default=None, help="Root directory containing motion vector HDF5 chunks")
    parser.add_argument("--optical-flow-root", type=str, default=None, help="Root directory containing CoTracker3 optical flow pickles")
    parser.add_argument("--dataset-fraction", type=float, default=1.0, help="Fraction of motion-vector chunks to load (0-1]")
    parser.add_argument("--disable-visibility", action="store_true", help="Ignore optical flow visibility mask when computing losses")
    parser.add_argument("--output-dir", default="mv_refiner_logs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--predict-residual", action="store_true", help="Predict residual added to motion vectors")
    parser.add_argument("--log-dataset", action="store_true", help="Print dataset loading progress and stats")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="mv_refiner", help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", default=None, help="Weights & Biases entity/team")
    parser.add_argument("--model-type", default="cnn", choices=["cnn", "mamba", "transformer"], help="Model architecture: cnn or mamba")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension for model")
    parser.add_argument("--transformer-num-layers", type=int, default=2, help="Number of transformer encoder layers (model-type transformer)")
    parser.add_argument("--transformer-num-heads", type=int, default=4, help="Number of attention heads for transformer refiner")
    parser.add_argument("--transformer-dropout", type=float, default=0.1, help="Dropout probability for transformer refiner")
    parser.add_argument("--transformer-ffn-multiplier", type=float, default=4.0, help="Feedforward width multiplier for transformer refiner")
    parser.add_argument("--mamba-d-state", type=int, default=16, help="Mamba state dimension")
    parser.add_argument("--mamba-d-conv", type=int, default=4, help="Mamba convolution dimension")
    parser.add_argument("--mamba-expand", type=int, default=2, help="Mamba expansion factor")

    args = parser.parse_args()

    if args.pkl_paths is None and (args.mv_root is None or args.optical_flow_root is None):
        parser.error("Provide --pkl-paths or both --mv-root and --optical-flow-root")
    if args.pkl_paths is None:
        args.pkl_paths = []

    if not (0.0 < args.dataset_fraction <= 1.0):
        parser.error("--dataset-fraction must be in (0, 1] range")

    if not 0.0 < args.val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1")

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_loader, val_loader = create_dataloaders(args)

    # Select model based on model type
    if args.model_type == "mamba":
        if Mamba is None:
            raise RuntimeError("Mamba module not available. Use --model-type cnn instead.")
        model = MambaMotionVectorRefiner(
            hidden_dim=args.hidden_dim,
            d_state=args.mamba_d_state,
            d_conv=args.mamba_d_conv,
            expand=args.mamba_expand,
        )
        print(f"[setup] Using Mamba-based refiner with hidden_dim={args.hidden_dim}")
    elif args.model_type == "transformer":
        model = TransformerMotionVectorRefiner(
            hidden_dim=args.hidden_dim,
            num_layers=args.transformer_num_layers,
            num_heads=args.transformer_num_heads,
            ffn_multiplier=args.transformer_ffn_multiplier,
            dropout=args.transformer_dropout,
        )
        print(f"[setup] Using transformer-based refiner with hidden_dim={args.hidden_dim}, layers={args.transformer_num_layers}, heads={args.transformer_num_heads}")
    else:
        model = MotionVectorRefiner(hidden_dim=args.hidden_dim)
        print(f"[setup] Using CNN-based refiner with hidden_dim={args.hidden_dim}")

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_epe = float("inf")
    history = []

    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb requested but not installed. `pip install wandb` inside the environment.")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
        )


    for epoch in range(1, args.epochs + 1):
        train_loss, train_epe = train_epoch(model, train_loader, optimizer, device, args)
        val_loss, val_epe = evaluate(model, val_loader, device, args)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_epe": train_epe,
                "val_loss": val_loss,
                "val_epe": val_epe,
            }
        )

        log_payload = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/epe": train_epe,
            "val/loss": val_loss,
            "val/epe": val_epe,
            "best_val_epe": best_val_epe,
        }

        print(
            f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | train_epe {train_epe:.4f} | "
            f"val_loss {val_loss:.4f} | val_epe {val_epe:.4f}"
        )

        if args.wandb:
            wandb.log(log_payload, step=epoch)

        if val_epe < best_val_epe:
            best_val_epe = val_epe
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_epe": val_epe,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")

        torch.save(model.state_dict(), output_dir / "last_model.pt")

    with (output_dir / "training_metrics.json").open("w") as f:
        json.dump(history, f, indent=2)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()


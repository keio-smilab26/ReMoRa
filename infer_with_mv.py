"""Codec-aware inference with the ReMoRa model.

Decodes motion vectors and frame types via PyAV in a single pass, then
drives ``GOPVideoLoader`` -> ``model.generate(..., modalities=["gop_video"])``.

CLI smoke test:

    uv run python infer_with_mv.py \\
        --video /path/to/video.mp4 \\
        --prompt "Describe what happens in this video."
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REMORA_ROOT = Path(__file__).resolve().parent

if str(REMORA_ROOT) not in sys.path:
    sys.path.insert(0, str(REMORA_ROOT))

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402


def _patch_causal_conv1d() -> None:
    import re
    import causal_conv1d_cuda

    if getattr(causal_conv1d_cuda, "_remora_compat_patched", False):
        return
    raw_fwd = causal_conv1d_cuda.causal_conv1d_fwd
    n_args = len(re.findall(r"arg\d+:", raw_fwd.__doc__ or ""))

    if n_args >= 8:
        def shim(*args, **kwargs):
            if len(args) == 7 and isinstance(args[-1], bool):
                x, weight, bias, seq_idx, initial_states, _drop, silu = args
                out = torch.empty_like(x)
                raw_fwd(x, weight, bias, seq_idx, initial_states, out, None, silu)
                return out
            return raw_fwd(*args, **kwargs)

        causal_conv1d_cuda.causal_conv1d_fwd = shim
    causal_conv1d_cuda._remora_compat_patched = True


_patch_causal_conv1d()

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # type: ignore  # noqa: E402
from llava.conversation import conv_templates  # type: ignore  # noqa: E402
from llava.mm_utils import tokenizer_image_token  # type: ignore  # noqa: E402
from llava.model.builder import load_pretrained_model  # type: ignore  # noqa: E402
from llava.train.gop_video_loader import GOPVideoLoader  # type: ignore  # noqa: E402


# 0=I, 1=P, 2=B (matches MotionVectorLoader); other PyAV picture types fall through to P.
_PICT_AV_TO_FRAME_TYPE = {1: 0, 2: 1, 3: 2}


@dataclass
class CodecData:
    motion_vectors: np.ndarray  # (T, H/block, W/block, 2) float32
    frame_types: np.ndarray  # (T,) int32 in {0,1,2}
    block_size: int
    width: int
    height: int


def _extract_codec_pyav(
    video_path: Path,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    import av
    from av.codec.context import Flags2
    from av.sidedata.motionvectors import MotionVectors

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.codec_context.flags2 |= Flags2.export_mvs
        stream.thread_type = "AUTO"
        width = int(stream.width)
        height = int(stream.height)

        mv_chunks: List[np.ndarray] = []
        frame_idxs: List[np.ndarray] = []
        frame_types: List[int] = []

        for i, frame in enumerate(container.decode(stream)):
            frame_types.append(_PICT_AV_TO_FRAME_TYPE.get(int(frame.pict_type), 1))
            for sd in frame.side_data:
                if isinstance(sd, MotionVectors):
                    arr = sd.to_ndarray()
                    if arr.size > 0:
                        mv_chunks.append(arr)
                        frame_idxs.append(np.full(len(arr), i + 1, dtype=np.int32))
                    break
    finally:
        container.close()

    ft_array = np.asarray(frame_types, dtype=np.int32)
    if not mv_chunks:
        return np.zeros((0, 8), dtype=np.float32), ft_array, width, height

    big_mv = np.concatenate(mv_chunks)
    big_frame = np.concatenate(frame_idxs)

    table = np.empty((big_mv.shape[0], 8), dtype=np.float32)
    table[:, 0] = big_frame
    table[:, 1] = big_mv["w"]
    table[:, 2] = big_mv["h"]
    table[:, 3] = big_mv["dst_x"]
    table[:, 4] = big_mv["dst_y"]
    table[:, 5] = big_mv["motion_x"]
    table[:, 6] = big_mv["motion_y"]
    table[:, 7] = big_mv["motion_scale"]
    return table, ft_array, width, height


def _vectorized_grid(
    table: np.ndarray, n_frames: int, width: int, height: int, block_size: int
) -> np.ndarray:
    grid_h = max(height // block_size, 1)
    grid_w = max(width // block_size, 1)
    cell_count = n_frames * grid_h * grid_w
    if table.shape[0] == 0 or cell_count == 0:
        return np.zeros((max(n_frames, 1), grid_h, grid_w, 2), dtype=np.float32)

    frame = table[:, 0].astype(np.int64) - 1
    block_w = table[:, 1].astype(np.int64)
    block_h = table[:, 2].astype(np.int64)
    dst_x = table[:, 3].astype(np.int64)
    dst_y = table[:, 4].astype(np.int64)
    motion_x = table[:, 5]
    motion_y = table[:, 6]
    motion_scale = table[:, 7].astype(np.int64)

    valid = (
        (frame >= 0)
        & (frame < n_frames)
        & (dst_x >= 0)
        & (dst_y >= 0)
        & (dst_x + block_w <= width)
        & (dst_y + block_h <= height)
    )
    if not valid.any():
        return np.zeros((n_frames, grid_h, grid_w, 2), dtype=np.float32)

    frame = frame[valid]
    block_w = block_w[valid]
    block_h = block_h[valid]
    dst_x = dst_x[valid]
    dst_y = dst_y[valid]
    scale_div = (1 << motion_scale[valid]).astype(np.float32)
    mx = (motion_x[valid] / scale_div).astype(np.float32)
    my = (motion_y[valid] / scale_div).astype(np.float32)

    cell_h = ((dst_y % block_size) + block_h + block_size - 1) // block_size
    cell_w = ((dst_x % block_size) + block_w + block_size - 1) // block_size

    n = frame.shape[0]
    max_ch = int(cell_h.max())
    max_cw = int(cell_w.max())
    y0 = dst_y // block_size
    x0 = dst_x // block_size

    dy = np.arange(max_ch, dtype=np.int64)
    dx = np.arange(max_cw, dtype=np.int64)
    valid_cells = (dy[None, :, None] < cell_h[:, None, None]) & (
        dx[None, None, :] < cell_w[:, None, None]
    )
    yy = np.clip(y0[:, None, None] + dy[None, :, None], 0, grid_h - 1)
    xx = np.clip(x0[:, None, None] + dx[None, None, :], 0, grid_w - 1)
    yy_b = np.broadcast_to(yy, (n, max_ch, max_cw))
    xx_b = np.broadcast_to(xx, (n, max_ch, max_cw))
    ff_b = np.broadcast_to(frame[:, None, None], (n, max_ch, max_cw))
    idx_full = ((ff_b * grid_h + yy_b) * grid_w + xx_b).ravel()

    keep = valid_cells.ravel()
    idx_keep = idx_full[keep]
    wx = np.broadcast_to(mx[:, None, None], (n, max_ch, max_cw)).reshape(-1)[keep]
    wy = np.broadcast_to(my[:, None, None], (n, max_ch, max_cw)).reshape(-1)[keep]

    sums_x = np.bincount(idx_keep, weights=wx, minlength=cell_count)
    sums_y = np.bincount(idx_keep, weights=wy, minlength=cell_count)
    counts = np.bincount(idx_keep, minlength=cell_count)

    sums = np.stack(
        [
            sums_x.reshape(n_frames, grid_h, grid_w).astype(np.float32),
            sums_y.reshape(n_frames, grid_h, grid_w).astype(np.float32),
        ],
        axis=-1,
    )
    counts = counts.reshape(n_frames, grid_h, grid_w)
    out = np.zeros_like(sums)
    nz = counts > 0
    out[nz] = sums[nz] / counts[nz, None]
    return out


def extract_codec_data(video_path: str | Path, block_size: int = 16) -> CodecData:
    """Extract per-frame motion vectors and frame types from a video.

    ``block_size`` is the MV-aggregation grid cell (not the codec partition
    size); 16 produces a 24x24 grid at 384x384.
    """
    video_path = Path(video_path)

    table, frame_types, width, height = _extract_codec_pyav(video_path)
    n_frames = max(
        int(table[:, 0].max()) if table.shape[0] else 0, len(frame_types)
    )
    mvs = _vectorized_grid(table, n_frames, width, height, block_size)

    n = min(mvs.shape[0], len(frame_types))
    if mvs.shape[0] > n:
        mvs = mvs[:n]
    if len(frame_types) > n:
        frame_types = frame_types[:n]
    if len(frame_types) < n:
        # Trailing frames PyAV decoded without MV side-data — pad as P.
        frame_types = np.concatenate(
            [frame_types, np.ones(n - len(frame_types), dtype=np.int32)]
        )

    return CodecData(
        motion_vectors=mvs,
        frame_types=frame_types,
        block_size=block_size,
        width=width,
        height=height,
    )


class InMemoryMVLoader:
    """Subset of MotionVectorLoader that GOPVideoLoader uses, backed by in-memory data."""

    def __init__(self, codec_data: CodecData):
        self._data = codec_data
        self._gop_structure_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def for_video(cls, video_path: str | Path, block_size: int = 16) -> "InMemoryMVLoader":
        return cls(extract_codec_data(video_path, block_size=block_size))

    def _gop_structure(self) -> Dict[str, Any]:
        if self._gop_structure_cache is not None:
            return self._gop_structure_cache
        ft = self._data.frame_types
        i_pos = np.where(ft == 0)[0]
        gops: List[Dict[str, int]] = []
        for k in range(len(i_pos)):
            start = int(i_pos[k])
            end = int(i_pos[k + 1] if k + 1 < len(i_pos) else len(ft))
            seg = ft[start:end]
            gops.append(
                {
                    "start": start,
                    "end": end,
                    "length": end - start,
                    "i_frames": int((seg == 0).sum()),
                    "p_frames": int((seg == 1).sum()),
                    "b_frames": int((seg == 2).sum()),
                }
            )
        self._gop_structure_cache = {
            "total_frames": int(len(ft)),
            "num_gops": int(len(i_pos)),
            "i_frame_positions": i_pos.tolist(),
            "gops": gops,
            "frame_type_counts": {
                "I": int((ft == 0).sum()),
                "P": int((ft == 1).sum()),
                "B": int((ft == 2).sum()),
            },
        }
        return self._gop_structure_cache

    def get_motion_vectors_with_types(
        self, video_path: str, fps: int = 16, retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        return {
            "motion_vectors": self._data.motion_vectors,
            "frame_types": self._data.frame_types,
            "gop_structure": self._gop_structure(),
        }


@dataclass
class GenerateConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    repetition_penalty: float = 1.1
    max_i_frames: int = 64
    conv_template: str = "qwen_1_5"


class ReMoRaCodecRunner:
    """Holds a loaded ReMoRa model + a per-video MV cache."""

    def __init__(
        self,
        model_path: str,
        base_model: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
        model_name: str = "llava_qwen_lora",
    ):
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=base_model,
            model_name=model_name,
            torch_dtype="bfloat16",
            device_map="auto",
            attn_implementation=None,
        )
        self.model.eval()
        self._mv_cache: Dict[str, InMemoryMVLoader] = {}

    def prepare_video(self, video_path: str | Path, block_size: int = 16) -> None:
        key = str(Path(video_path).resolve())
        if key in self._mv_cache:
            return
        self._mv_cache[key] = InMemoryMVLoader.for_video(video_path, block_size=block_size)

    def has_video(self, video_path: str | Path) -> bool:
        return str(Path(video_path).resolve()) in self._mv_cache

    def _build_gop_pack(
        self, video_path: Path, max_i_frames: int
    ) -> Tuple[Dict[str, Any], "VideoReader"]:
        key = str(video_path.resolve())
        if key not in self._mv_cache:
            self.prepare_video(video_path)
        mv_loader = self._mv_cache[key]
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        # Use the video's native fps so MV indices and frame indices line up
        # (otherwise _mv_index_to_video_index drifts).
        native_fps = max(int(round(vr.get_avg_fps())), 1)
        gop_loader = GOPVideoLoader(
            motion_vector_loader=mv_loader,
            max_i_frames=max_i_frames,
            max_pb_per_i=32,
        )
        gop_data = gop_loader.load_gop_video(
            str(video_path),
            vr,
            fps=native_fps,
            uniform_sample=True,
            fallback_to_frames=True,
        )
        if gop_data is None:
            raise RuntimeError("GOP loading failed (and no fallback was usable)")
        hybrid = gop_loader.create_hybrid_tensor(gop_data, self.image_processor)
        if hybrid is None:
            raise RuntimeError("create_hybrid_tensor returned None")
        hybrid["_video_time"] = gop_data.video_time
        hybrid["_num_i_frames"] = len(gop_data.i_frames)
        return hybrid, vr

    def _per_gop_mv_maps(
        self,
        motion_vectors: torch.Tensor,
        gop_boundaries: List[Tuple[int, int]],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if motion_vectors.numel() == 0:
            return None
        mv = motion_vectors.to(device)
        mags = (mv[..., 0] ** 2 + mv[..., 1] ** 2).sqrt()
        per: List[torch.Tensor] = []
        for start, end in gop_boundaries:
            if end > start:
                per.append(mags[start:end].mean(dim=0))
            else:
                per.append(mags[start : start + 1].mean(dim=0))
        if not per:
            return None
        return torch.stack(per, dim=0)

    def _build_prompt(
        self,
        prompt: str,
        num_i_frames: int,
        video_time: float,
        conv_template: str,
    ) -> str:
        time_inst = (
            f"The video lasts for {video_time:.2f} seconds, and "
            f"{num_i_frames} I-frames are uniformly sampled from it.\n"
            "Please answer the following questions related to this video."
        )
        question = DEFAULT_IMAGE_TOKEN + f"{time_inst}\n{prompt}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def generate(
        self,
        video_path: str | Path,
        prompt: str,
        cfg: GenerateConfig = GenerateConfig(),
        streamer: Any = None,
    ) -> str:
        video_path = Path(video_path)
        hybrid, _ = self._build_gop_pack(video_path, cfg.max_i_frames)

        device = self.model.device
        i_frames = hybrid["i_frames"].to(device).bfloat16()
        images = [i_frames]

        core = self.model.get_model()
        core.pending_mv_seqs = [hybrid["motion_vectors"].to(device).bfloat16()]
        core.pending_i_indices = [hybrid["i_frame_indices"].to(device)]
        core.pending_gop_boundaries = [hybrid["gop_boundaries"]]

        prompt_text = self._build_prompt(
            prompt,
            hybrid["_num_i_frames"],
            hybrid["_video_time"],
            cfg.conv_template,
        )
        input_ids = (
            tokenizer_image_token(
                prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(device)
        )

        gen_kwargs: Dict[str, Any] = dict(
            inputs=input_ids,
            images=images,
            modalities=["gop_video"],
            do_sample=cfg.temperature > 0,
            temperature=max(cfg.temperature, 1e-5),
            max_new_tokens=cfg.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=cfg.repetition_penalty,
        )
        if streamer is not None:
            gen_kwargs["streamer"] = streamer

        with torch.inference_mode():
            out = self.model.generate(**gen_kwargs)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to a video file")
    ap.add_argument(
        "--prompt",
        default="Describe what happens in this video in two sentences.",
    )
    ap.add_argument(
        "--checkpoint",
        default=str(REMORA_ROOT / "checkpoints" / "ReMoRa-7B"),
        help="ReMoRa LoRA checkpoint directory",
    )
    ap.add_argument(
        "--base",
        default="lmms-lab/LLaVA-Video-7B-Qwen2",
        help="Base model the LoRA was trained against",
    )
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-i-frames", type=int, default=64)
    args = ap.parse_args()

    runner = ReMoRaCodecRunner(model_path=args.checkpoint, base_model=args.base)
    cfg = GenerateConfig(
        max_new_tokens=args.max_new_tokens, max_i_frames=args.max_i_frames
    )
    out = runner.generate(args.video, args.prompt, cfg)
    print(out)


if __name__ == "__main__":
    main()

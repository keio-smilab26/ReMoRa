"""Batch-extract motion vectors + I/P/B frame types into HDF5 shards.

Output layout:

    <output_dir>/mv_shard_0000_16fps.h5
        data/<video_stem>             float32 (T, H/bs, W/bs, 2)  attr original_path
        data/<video_stem>_frame_types int32   (T,)  0=I, 1=P, 2=B
    <output_dir>/master_index.json    built afterwards from the H5 files

Usage:
    python scripts/extract_motion_vectors.py \\
        --video-root /path/to/videos \\
        --output-dir DATAS/motion_vectors \\
        --fps 16 --block-size 16

Then pass the directory to training via --motion_vector_dir
(or set REMORA_MV_DIR). Pass --index-only to skip extraction and just
(re)build master_index.json from existing shards.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_with_mv import extract_codec_data  # noqa: E402
from llava.train.create_video_index import build_index_from_hdf5_files  # noqa: E402


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}


def iter_videos(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            yield path


def shard_path(output_dir: Path, shard_idx: int, fps: int) -> Path:
    return output_dir / f"mv_shard_{shard_idx:04d}_{fps}fps.h5"


def write_one(h5f: h5py.File, dataset_name: str, mvs: np.ndarray, frame_types: np.ndarray, original_path: str, fps: int) -> None:
    group = h5f.require_group("data")
    if dataset_name in group:
        del group[dataset_name]
    if f"{dataset_name}_frame_types" in group:
        del group[f"{dataset_name}_frame_types"]
    ds = group.create_dataset(dataset_name, data=mvs.astype(np.float32), compression="gzip", compression_opts=4)
    ds.attrs["original_path"] = original_path
    ds.attrs["fps"] = fps
    group.create_dataset(f"{dataset_name}_frame_types", data=frame_types.astype(np.int32), compression="gzip", compression_opts=4)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract motion vectors + frame types into HDF5 shards.")
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--videos-per-shard", type=int, default=500)
    ap.add_argument("--index-only", action="store_true", help="Skip extraction; just rebuild master_index.json.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "master_index.json"

    if not args.index_only:
        if not args.video_root.is_dir():
            raise FileNotFoundError(f"video-root not found: {args.video_root}")

        shard_idx = 0
        in_shard = 0
        h5f: h5py.File | None = None
        for video in iter_videos(args.video_root):
            if h5f is None or in_shard >= args.videos_per_shard:
                if h5f is not None:
                    h5f.close()
                h5f = h5py.File(shard_path(args.output_dir, shard_idx, args.fps), "w")
                shard_idx += 1
                in_shard = 0
            try:
                codec = extract_codec_data(video, block_size=args.block_size)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {video}: {exc}", flush=True)
                continue
            write_one(h5f, video.stem, codec.motion_vectors, codec.frame_types, str(video), args.fps)
            in_shard += 1
        if h5f is not None:
            h5f.close()

    build_index_from_hdf5_files(args.output_dir, index_path)
    print(f"wrote {index_path}")


if __name__ == "__main__":
    main()

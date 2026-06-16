"""
Builds a lightweight index mapping original video paths (or names) to
their corresponding datasets inside split HDF5 motion-vector files.

The index schema matches what MotionVectorLoader expects:

{
  "<video_path_or_name>": {
    "4fps": {"file": "relative/path/to/file.h5", "dataset": "<ds_name>"},
    "8fps": {"file": "...", "dataset": "..."},
    "16fps": {"file": "...", "dataset": "..."}
  },
  ...
}

Notes:
- We detect FPS from the filename (containing 4fps/8fps/16fps). If not found,
  the file is skipped.
- We look under the "data" group if present; otherwise, we inspect root.
- We skip keys that end with "_frame_types" since those are auxiliary arrays.
- If a dataset has an attribute like "original_path" or "video_path", we use it
  as the index key; otherwise, we fall back to the dataset name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union
import json
import h5py


def _detect_fps_label_from_filename(path: Path) -> str | None:
    name = path.as_posix().lower()
    if "4fps" in name:
        return "4fps"
    if "8fps" in name:
        return "8fps"
    if "16fps" in name:
        return "16fps"
    return None


def _iter_datasets(h5f: h5py.File):
    # Prefer a top-level group named "data" if present
    group = h5f["data"] if "data" in h5f else h5f

    for key in group.keys():
        # Skip frame type auxiliaries
        if str(key).endswith("_frame_types"):
            continue
        obj = group.get(key, None)
        if obj is None:
            continue
        # Consider only datasets (not groups)
        if isinstance(obj, h5py.Dataset):
            yield str(key), obj


def _get_original_path(ds: h5py.Dataset, default_name: str) -> str:
    # Common attribute names that may hold the source video path
    for attr in ("original_path", "video_path", "source_path"):
        if attr in ds.attrs:
            val = ds.attrs[attr]
            if isinstance(val, (bytes, bytearray)):
                try:
                    return val.decode("utf-8", errors="ignore")
                except Exception:
                    pass
            return str(val)
    return default_name


def build_index_from_hdf5_files(hdf5_dir: Union[str, Path], index_path: Union[str, Path]) -> Dict:
    hdf5_dir = Path(hdf5_dir)
    index_path = Path(index_path)

    if not hdf5_dir.exists():
        raise FileNotFoundError(f"HDF5 directory not found: {hdf5_dir}")

    index: Dict[str, Dict] = {}

    # Find all HDF5 files recursively
    files = sorted(hdf5_dir.rglob("*.h5"))
    for h5_file in files:
        rel_file = h5_file.relative_to(hdf5_dir).as_posix()
        try:
            with h5py.File(h5_file, "r") as h5f:
                for ds_name, ds in _iter_datasets(h5f):
                    # Determine FPS per-dataset; prefer dataset attribute, then filename
                    fps_label = None
                    try:
                        fps_attr = ds.attrs.get("fps")
                        if fps_attr is not None:
                            fps_val = float(fps_attr)
                            if int(round(fps_val)) in (4, 8, 16):
                                fps_label = f"{int(round(fps_val))}fps"
                    except Exception:
                        pass
                    if fps_label is None:
                        fps_label = _detect_fps_label_from_filename(h5_file)
                    if fps_label is None:
                        # Unknown FPS: skip this dataset
                        continue

                    key = _get_original_path(ds, ds_name)
                    entry = index.setdefault(key, {})
                    entry[fps_label] = {"file": rel_file, "dataset": ds_name}
        except Exception:
            # Best-effort: continue scanning other files
            continue

    # Persist the index for future runs
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
    except Exception:
        # If writing fails, still return the in-memory index
        pass

    return index

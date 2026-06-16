"""
Utility to easily load motion vectors from split HDF5 files.
Handles the complexity of finding videos across multiple files and renamed duplicates.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotionVectorLoader:
    """Load motion vectors from split HDF5 files with automatic indexing."""
    
    def __init__(self, hdf5_dir: str, index_path: str = None):
        """
        Initialize loader.
        
        Args:
            hdf5_dir: Directory containing HDF5 files
            index_path: Path to index JSON (will be created if doesn't exist)
        """
        self.hdf5_dir = Path(hdf5_dir)
        # Default index filename; accept a few common alternatives
        self.index_path = Path(index_path) if index_path else self.hdf5_dir / 'video_index.json'
        self.index = None
        self._h5_cache = {}  # Cache open HDF5 files
        
        # Load or build index
        self._load_or_build_index()

    def _get_h5_file(self, h5_path: Path):
        """Return cached HDF5 file handle."""
        if h5_path not in self._h5_cache:
            self._h5_cache[h5_path] = h5py.File(h5_path, 'r')
        return self._h5_cache[h5_path]

    def _read_dataset(self, info: Dict, suffix: str = "") -> Optional[np.ndarray]:
        """
        Read a dataset from the HDF5 file referenced by the video index.

        Args:
            info: Metadata entry from the video index.
            suffix: Optional dataset suffix (e.g., "_frame_types").

        Returns:
            Numpy array with dataset contents or None if unavailable.
        """
        h5_path = self.hdf5_dir / info['file']
        try:
            h5f = self._get_h5_file(h5_path)
            dataset_name = info['dataset'] + suffix

            # Don't add "data/" prefix if dataset_name already starts with it
            if dataset_name.startswith("data/"):
                dataset_path = dataset_name
            else:
                dataset_path = f"data/{dataset_name}" if "data" in h5f else dataset_name

            if dataset_path in h5f:
                return np.array(h5f[dataset_path])
            logger.warning(f"Dataset not found: {dataset_path} in {info['file']}")
        except Exception as e:
            logger.error(f"Error loading dataset {info.get('dataset')} from {info.get('file')}: {e}")
        return None

    @staticmethod
    def _fps_key_to_int(fps_key: str) -> Optional[int]:
        """Convert an index key like '8fps' to an integer FPS value."""
        try:
            return int(fps_key.replace('fps', ''))
        except ValueError:
            return None

    def _downsample_from_base(
        self,
        video_info: Dict,
        target_fps_key: str,
        base_fps: int = 16,
        suffix: str = ""
    ) -> Optional[np.ndarray]:
        """
        Downsample 16fps data to match a lower target FPS when direct data is unavailable.
        """
        target_fps = self._fps_key_to_int(target_fps_key)
        if target_fps is None or target_fps >= base_fps:
            return None

        base_key = f"{base_fps}fps"
        if base_key not in video_info:
            return None

        factor = base_fps // target_fps
        if base_fps % target_fps != 0:
            logger.warning(
                f"Cannot downsample {base_fps}fps data to non-divisible target {target_fps}fps"
            )
            return None

        base_info = video_info[base_key]
        base_data = self._read_dataset(base_info, suffix=suffix)
        if base_data is None:
            return None

        # Simple stride-based downsampling per user request.
        return base_data[::factor]
    
    def _load_or_build_index(self):
        """Load existing index or build new one."""
        if self.index_path.exists():
            logger.info(f"Loading index from {self.index_path}")
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            # Try common fallbacks before building
            for alt in ['master_index.json', 'index.json']:
                alt_path = self.hdf5_dir / alt
                if alt_path.exists():
                    logger.info(f"Loading index from {alt_path}")
                    with open(alt_path, 'r') as f:
                        self.index = json.load(f)
                    self.index_path = alt_path
                    break
            if self.index is None:
                logger.info("Building new index...")
                self._build_index()
    
    def _build_index(self):
        """Build index by scanning all HDF5 files."""
        try:
            # Prefer local package-relative implementation
            from .create_video_index import build_index_from_hdf5_files
        except Exception:
            # Fallback to top-level module if available
            from create_video_index import build_index_from_hdf5_files  # type: ignore
        self.index = build_index_from_hdf5_files(self.hdf5_dir, self.index_path)
    
    def get_motion_vectors(self, video_path: str, fps: int = None) -> Dict[str, np.ndarray]:
        """
        Get motion vectors for a video.
        
        Args:
            video_path: Original video path or filename
            fps: Specific FPS to load (4, 8, or 16). If None, returns all.
        
        Returns:
            Dictionary with fps as keys and motion vectors as values
        """
        # Find video in index
        video_info = self._find_video(video_path)
        if not video_info:
            logger.error(f"Video not found: {video_path}")
            return {}
        
        results = {}
        fps_keys = [f"{fps}fps"] if fps else ['4fps', '8fps', '16fps']
        
        for fps_key in fps_keys:
            info = video_info.get(fps_key)
            data = self._read_dataset(info, suffix="") if info else None

            if data is None:
                downsampled = self._downsample_from_base(video_info, fps_key, suffix="")
                if downsampled is not None:
                    logger.info(
                        f"Downsampled 16fps motion vectors to {fps_key} for {video_path}"
                    )
                    results[fps_key] = downsampled
                    continue
                if info is None:
                    logger.warning(f"FPS {fps_key} not available for video: {video_path}")
                else:
                    logger.error(f"Failed to load {fps_key} motion vectors for {video_path}")
                continue

            results[fps_key] = data
        
        return results
    
    def get_frame_types(self, video_path: str, fps: int = 16) -> Optional[np.ndarray]:
        """
        Get frame type indices for a video.

        Args:
            video_path: Original video path or filename
            fps: FPS rate to get frame types for (default: 16)
        
        Returns:
            Array of frame types (0=I, 1=P, 2=B) or None if not found
        """
        video_info = self._find_video(video_path)
        if not video_info:
            logger.error(f"Video not found: {video_path}")
            return None
        
        fps_key = f"{fps}fps"
        info = video_info.get(fps_key)
        frame_types = self._read_dataset(info, suffix="_frame_types") if info else None

        if frame_types is not None:
            return frame_types

        # Attempt to downsample from 16fps if direct data unavailable
        downsampled = self._downsample_from_base(video_info, fps_key, suffix="_frame_types")
        if downsampled is not None:
            logger.info(
                f"Downsampled 16fps frame types to {fps_key} for {video_path}"
            )
            return downsampled

        logger.error(f"FPS {fps} not available for video: {video_path}")
        return None
    
    def get_gop_structure(self, video_path: str, fps: int = 16) -> Optional[Dict]:
        """
        Get GOP (Group of Pictures) structure for a video.

        Args:
            video_path: Original video path or filename
            fps: FPS rate to analyze (default: 16)
        
        Returns:
            Dictionary with GOP information or None if not found
        """
        frame_types = self.get_frame_types(video_path, fps)
        if frame_types is None:
            return None
        
        # Find I-frame positions
        i_frame_positions = np.where(frame_types == 0)[0]
        
        # Analyze GOP structure
        gops = []
        for i in range(len(i_frame_positions)):
            start = i_frame_positions[i]
            end = i_frame_positions[i+1] if i+1 < len(i_frame_positions) else len(frame_types)
            
            gop_frames = frame_types[start:end]
            gops.append({
                'start': int(start),
                'end': int(end),
                'length': int(end - start),
                'i_frames': int(np.sum(gop_frames == 0)),
                'p_frames': int(np.sum(gop_frames == 1)),
                'b_frames': int(np.sum(gop_frames == 2))
            })
        
        return {
            'total_frames': len(frame_types),
            'num_gops': len(i_frame_positions),
            'i_frame_positions': i_frame_positions.tolist(),
            'gops': gops,
            'frame_type_counts': {
                'I': int(np.sum(frame_types == 0)),
                'P': int(np.sum(frame_types == 1)),
                'B': int(np.sum(frame_types == 2))
            }
        }
    
    def get_motion_vectors_with_types(self, video_path: str, fps: int = 16, retry_count: int = 2) -> Optional[Dict]:
        """
        Get both motion vectors and frame types for a video.

        Args:
            video_path: Original video path or filename
            fps: FPS rate (default: 16)
            retry_count: Number of retries on failure (default: 2)

        Returns:
            Dictionary with 'motion_vectors' and 'frame_types' keys
        """
        for attempt in range(retry_count + 1):
            try:
                mvs = self.get_motion_vectors(video_path, fps)
                frame_types = self.get_frame_types(video_path, fps)

                fps_key = f"{fps}fps"
                if fps_key not in mvs or frame_types is None:
                    if attempt < retry_count:
                        logger.warning(f"Attempt {attempt + 1} failed for {video_path}, retrying...")
                        # Clear cache in case of stale file handles
                        if any(video_path in str(p) for p in self._h5_cache.keys()):
                            for path in list(self._h5_cache.keys()):
                                if video_path in str(path):
                                    self._h5_cache[path].close()
                                    del self._h5_cache[path]
                        continue
                    return None

                return {
                    'motion_vectors': mvs[fps_key],
                    'frame_types': frame_types,
                    'gop_structure': self.get_gop_structure(video_path, fps)
                }
            except Exception as e:
                logger.warning(f"Error on attempt {attempt + 1} for {video_path}: {e}")
                if attempt >= retry_count:
                    return None
        return None
    
    def _find_video(self, video_path: str) -> Optional[Dict]:
        """Find video in index."""
        # Exact match
        if video_path in self.index:
            return self.index[video_path]
        
        # Try by filename
        video_name = Path(video_path).name
        for path, info in self.index.items():
            if Path(path).name == video_name:
                return info
        
        # Try by stem (without extension)
        video_stem = Path(video_path).stem
        for path, info in self.index.items():
            if Path(path).stem == video_stem:
                return info
        
        return None
    
    def list_videos(self) -> List[str]:
        """List all available video paths."""
        return list(self.index.keys())
    
    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """Get metadata about a video."""
        return self._find_video(video_path)
    
    def close(self):
        """Close cached HDF5 files."""
        for h5f in self._h5_cache.values():
            h5f.close()
        self._h5_cache.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def quick_load(hdf5_dir: str, video_path: str, fps: int = None) -> Dict[str, np.ndarray]:
    """
    Quick function to load motion vectors without creating loader instance.
    
    Example:
        mvs = quick_load('/path/to/hdf5s', 'video.mp4', fps=8)
    """
    with MotionVectorLoader(hdf5_dir) as loader:
        return loader.get_motion_vectors(video_path, fps)


def main():
    """Example usage and testing."""
    import argparse
    parser = argparse.ArgumentParser(description="Load motion vectors from split HDF5 files")
    parser.add_argument('hdf5_dir', help='Directory with HDF5 files')
    parser.add_argument('--video', help='Video to load')
    parser.add_argument('--fps', type=int, help='Specific FPS (4, 8, or 16)')
    parser.add_argument('--list', action='store_true', help='List all videos')
    
    args = parser.parse_args()
    
    with MotionVectorLoader(args.hdf5_dir) as loader:
        if args.list:
            videos = loader.list_videos()
            print(f"Found {len(videos)} videos:")
            for v in videos[:10]:
                print(f"  - {v}")
            if len(videos) > 10:
                print(f"  ... and {len(videos)-10} more")
        
        elif args.video:
            mvs = loader.get_motion_vectors(args.video, args.fps)
            if mvs:
                for fps_key, data in mvs.items():
                    print(f"{fps_key}: shape={data.shape}, dtype={data.dtype}")
            else:
                print(f"Video not found: {args.video}")
        
        else:
            # Show example
            videos = loader.list_videos()
            if videos:
                print(f"Example - loading first video: {videos[0]}")
                mvs = loader.get_motion_vectors(videos[0])
                for fps_key, data in mvs.items():
                    print(f"  {fps_key}: shape={data.shape}")


if __name__ == "__main__":
    main()

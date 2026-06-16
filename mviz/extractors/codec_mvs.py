#!/usr/bin/env python3
"""
Extract motion vectors from video codecs using FFmpeg's extract_mvs.

This module provides functionality to:
1. Extract raw motion vectors from H.264/H.265 video streams
2. Convert them to numpy arrays suitable for deep learning
3. Support multiple block sizes (4x4, 8x8, 16x16)
"""

import subprocess
import numpy as np
import json
from pathlib import Path
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodecMotionVectorExtractor:
    """Extract actual codec motion vectors from video files."""
    
    def __init__(self, extract_mvs_path: str = './extract_mvs_ffmpeg'):
        """
        Initialize the codec motion vector extractor.
        
        Args:
            extract_mvs_path: Path to the compiled extract_mvs binary
        """
        self.extract_mvs_path = Path(extract_mvs_path)
        if not self.extract_mvs_path.exists():
            raise FileNotFoundError(
                f"extract_mvs_ffmpeg not found at {self.extract_mvs_path}. "
                "Please compile it from src/extractors/extract_mvs_ffmpeg.c"
            )
    
    def extract(self, video_path: str, output_dir: str = 'outputs/features/motion_vectors',
                block_sizes: List[int] = [4, 8, 16]) -> Dict:
        """
        Extract motion vectors from a video file.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted motion vectors
            block_sizes: List of block sizes to generate (e.g., [4, 8, 16])
            
        Returns:
            Dictionary containing extraction results and metadata
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract raw motion vectors using FFmpeg
        logger.info(f"Extracting motion vectors from: {video_path}")
        raw_mvs = self._extract_raw_motion_vectors(video_path)
        
        if not raw_mvs:
            logger.error("No motion vectors extracted")
            return {}
        
        # Step 2: Analyze video dimensions and block sizes
        video_info = self._analyze_video_properties(raw_mvs, video_path)
        logger.info(f"Video dimensions: {video_info['width']}x{video_info['height']}")
        logger.info(f"Codec block sizes: {video_info['block_sizes']}")
        
        # Step 3: Convert to numpy arrays for each block size
        results = {}
        for block_size in block_sizes:
            logger.info(f"\nProcessing {block_size}x{block_size} grid...")
            mv_array = self._convert_to_grid(raw_mvs, video_info, block_size)
            
            # Save numpy array
            np_path = output_dir / f'motion_vectors_{block_size}x{block_size}.npy'
            np.save(np_path, mv_array)
            
            # Save metadata
            metadata = {
                'shape': list(mv_array.shape),
                'dtype': str(mv_array.dtype),
                'block_size': block_size,
                'video_path': str(video_path),
                'video_dimensions': [video_info['width'], video_info['height']],
                'extraction_method': 'FFmpeg codec motion vectors',
                'description': f'Shape: (n_frames, h//{block_size}, w//{block_size}, 2)'
            }
            
            json_path = output_dir / f'motion_vectors_{block_size}x{block_size}.json'
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            results[f'{block_size}x{block_size}'] = {
                'array': mv_array,
                'path': str(np_path),
                'metadata': metadata
            }
            
            logger.info(f"Saved to {np_path}, shape: {mv_array.shape}")
        
        # Save summary
        summary = {
            'video_path': str(video_path),
            'total_vectors': len(raw_mvs),
            'video_info': video_info,
            'extracted_grids': list(results.keys()),
            'output_dir': str(output_dir)
        }
        
        summary_path = output_dir / 'extraction_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nExtraction complete. Files saved to {output_dir}")
        return results
    
    def _extract_raw_motion_vectors(self, video_path: Path) -> List[Dict]:
        """Extract raw motion vectors using the extract_mvs binary."""
        cmd = [str(self.extract_mvs_path), str(video_path)]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"extract_mvs failed: {result.stderr}")
            return []
        
        # Parse CSV output
        lines = result.stdout.strip().split('\n')
        motion_vectors = []
        
        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 12:
                    try:
                        mv = {
                            'frame_num': int(parts[0]),
                            'source': int(parts[1]),
                            'block_w': int(parts[2]),
                            'block_h': int(parts[3]),
                            'src_x': int(parts[4]),
                            'src_y': int(parts[5]),
                            'dst_x': int(parts[6]),
                            'dst_y': int(parts[7]),
                            'motion_x': int(parts[9]),
                            'motion_y': int(parts[10]),
                            'motion_scale': int(parts[11])
                        }
                        motion_vectors.append(mv)
                    except ValueError:
                        continue
        
        return motion_vectors
    
    def _analyze_video_properties(self, motion_vectors: List[Dict], video_path: Path) -> Dict:
        """Analyze video properties from motion vectors."""
        # Get actual video dimensions using cv2
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Use actual video dimensions instead of motion vector coordinates
        width = actual_width
        height = actual_height
        
        # Count block sizes used by codec
        block_sizes = defaultdict(int)
        for mv in motion_vectors:
            size = f"{mv['block_w']}x{mv['block_h']}"
            block_sizes[size] += 1
        
        # Get frame count
        frames_mv = defaultdict(list)
        for mv in motion_vectors:
            frames_mv[mv['frame_num']].append(mv)
        n_frames = max(frames_mv.keys()) if frames_mv else 0
        
        return {
            'width': width,
            'height': height,
            'n_frames': n_frames,
            'block_sizes': dict(block_sizes),
            'frames_mv': frames_mv
        }
    
    def _convert_to_grid(self, motion_vectors: List[Dict], video_info: Dict, 
                        block_size: int) -> np.ndarray:
        """Convert motion vectors to a regular grid."""
        width = video_info['width']
        height = video_info['height']
        n_frames = video_info['n_frames']
        frames_mv = video_info['frames_mv']
        
        # Calculate grid dimensions
        grid_h = height // block_size
        grid_w = width // block_size
        
        # Initialize output array
        mv_array = np.zeros((n_frames, grid_h, grid_w, 2), dtype=np.float32)
        
        # Process each frame
        for frame_num, mvs in frames_mv.items():
            if frame_num > n_frames:
                continue
            
            # Accumulate motion vectors
            mv_sum = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
            mv_count = np.zeros((grid_h, grid_w), dtype=int)
            
            for mv in mvs:
                # Skip motion vectors that extend beyond actual frame boundaries
                if (mv['dst_x'] >= width or mv['dst_y'] >= height or 
                    mv['dst_x'] + mv['block_w'] > width or mv['dst_y'] + mv['block_h'] > height):
                    continue
                    
                # Scale motion vectors
                scale = 1 << mv['motion_scale']
                motion_x = mv['motion_x'] / scale
                motion_y = mv['motion_y'] / scale
                
                # Map to target grid
                y1 = mv['dst_y'] // block_size
                y2 = (mv['dst_y'] + mv['block_h'] + block_size - 1) // block_size
                x1 = mv['dst_x'] // block_size
                x2 = (mv['dst_x'] + mv['block_w'] + block_size - 1) // block_size
                
                # Clip to grid bounds
                y1 = max(0, min(y1, grid_h - 1))
                y2 = max(0, min(y2, grid_h))
                x1 = max(0, min(x1, grid_w - 1))
                x2 = max(0, min(x2, grid_w))
                
                # Assign motion vector to overlapping blocks
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        mv_sum[y, x, 0] += motion_x
                        mv_sum[y, x, 1] += motion_y
                        mv_count[y, x] += 1
            
            # Average motion vectors
            mask = mv_count > 0
            mv_sum[mask, 0] /= mv_count[mask]
            mv_sum[mask, 1] /= mv_count[mask]
            
            # Store in array (frame_num is 1-indexed)
            if frame_num - 1 < n_frames:
                mv_array[frame_num - 1] = mv_sum
        
        return mv_array


def extract_motion_vectors(video_path: str, output_dir: Optional[str] = None,
                          block_sizes: List[int] = [4, 8, 16]) -> Dict:
    """
    Convenience function to extract motion vectors from a video.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory (default: outputs/features/motion_vectors)
        block_sizes: Block sizes to extract (default: [4, 8, 16])
        
    Returns:
        Dictionary with extracted motion vectors
    """
    # Look for extract_mvs_ffmpeg relative to this package
    import os
    import shutil
    package_root = Path(__file__).parent.parent
    extract_mvs_path = package_root / 'extract_mvs_ffmpeg'

    # Fall back to current directory, then PATH
    if not extract_mvs_path.exists():
        extract_mvs_path = Path('./extract_mvs_ffmpeg')
    if not extract_mvs_path.exists():
        path_binary = shutil.which('extract_mvs_ffmpeg')
        if path_binary:
            extract_mvs_path = Path(path_binary)
        else:
            raise FileNotFoundError(
                "extract_mvs_ffmpeg binary not found. Place it in the project root "
                "or add it to your PATH."
            )
    
    extractor = CodecMotionVectorExtractor(str(extract_mvs_path))
    return extractor.extract(video_path, output_dir or 'outputs/features/motion_vectors', 
                            block_sizes)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m extractors.codec_mvs <video_path> [block_sizes]")
        print("Example: python -m extractors.codec_mvs video.mp4 4,8,16")
        sys.exit(1)
    
    video_path = sys.argv[1]
    block_sizes = [4, 8, 16]
    
    if len(sys.argv) > 2:
        block_sizes = [int(x) for x in sys.argv[2].split(',')]
    
    results = extract_motion_vectors(video_path, block_sizes=block_sizes)
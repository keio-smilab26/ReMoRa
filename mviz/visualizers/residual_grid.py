#!/usr/bin/env python3
"""
Visualize motion vectors and residuals in a grid layout.

Creates visualizations showing:
- Original frames
- Motion vectors as colored blocks
- Residuals after motion compensation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import json
import logging
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotionResidualVisualizer:
    """Create visualizations of motion vectors and residuals."""
    
    def __init__(self, video_path: str):
        """Initialize visualizer with video file."""
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {video_path} - {self.width}x{self.height}, {self.frame_count} frames")
    
    def create_visualization(self, motion_vectors_path: str, 
                           output_path: Optional[str] = None,
                           num_frames: int = 5) -> str:
        """
        Create grid visualization of frames, motion vectors, and residuals.
        
        Args:
            motion_vectors_path: Path to .npy file with motion vectors
            output_path: Output path for visualization (optional)
            num_frames: Number of frames to visualize
            
        Returns:
            Path to saved visualization
        """
        # Load motion vectors
        mv_data = np.load(motion_vectors_path)
        mv_path = Path(motion_vectors_path)
        
        # Determine block size from filename or metadata
        block_size = self._get_block_size(mv_path)
        
        logger.info(f"Loaded motion vectors: {mv_data.shape}, block size: {block_size}")
        
        # Get frame types
        frame_types = self._get_frame_types()
        
        # Select frames to visualize
        frame_indices = self._select_frames(frame_types, num_frames, len(mv_data))
        
        # Read frames
        frames_data = self._read_frames(frame_indices, frame_types, mv_data, block_size)
        
        # Create visualization
        output_path = self._create_grid_visualization(frames_data, block_size, output_path)
        
        self.cap.release()
        return output_path
    
    def _get_block_size(self, mv_path: Path) -> int:
        """Extract block size from filename or metadata."""
        # Try to get from filename 
        # Supports formats: "motion_vectors_4x4.npy", "mv_4.npy", "motion_residuals_4x4.npy"
        stem = mv_path.stem
        
        # First try: look for pattern like "4x4"
        if 'x' in stem:
            try:
                # Split on 'x' first: "motion_vectors_4x4" -> ["motion_vectors_4", "4"]
                parts = stem.split('x')
                if len(parts) >= 2:
                    # Get the part before 'x': "motion_vectors_4" -> "4"
                    first_part = parts[0].split('_')[-1]
                    block_size = int(first_part)
                    logger.debug(f"Extracted block size {block_size} from filename (NxN format): {mv_path.name}")
                    return block_size
            except Exception as e:
                logger.warning(f"Failed to extract block size from NxN format in {mv_path.name}: {e}")
        
        # Second try: look for pattern like "mv_4" or ending with "_4"
        if '_' in stem:
            try:
                parts = stem.split('_')
                # Try the last part
                if parts[-1].isdigit():
                    block_size = int(parts[-1])
                    logger.debug(f"Extracted block size {block_size} from filename (underscore format): {mv_path.name}")
                    return block_size
            except Exception as e:
                logger.warning(f"Failed to extract block size from underscore format in {mv_path.name}: {e}")
        
        # Check for metadata file
        json_path = mv_path.with_suffix('.json')
        if json_path.exists():
            try:
                with open(json_path) as f:
                    metadata = json.load(f)
                    block_size = metadata.get('block_size', 4)  # Default to 4 instead of 16
                    logger.debug(f"Got block size {block_size} from metadata file")
                    return block_size
            except Exception as e:
                logger.warning(f"Failed to read metadata from {json_path}: {e}")
        
        # Default to 4 (more common for our use case)
        logger.warning(f"Using default block size 4 for {mv_path.name}")
        return 4  # Default changed from 16 to 4
    
    def _get_frame_types(self) -> List[Dict]:
        """Get frame type information using ffprobe."""
        import subprocess
        
        cmd = [
            'ffprobe',
            '-select_streams', 'v:0',
            '-show_entries', 'frame=pict_type,coded_picture_number,key_frame',
            '-of', 'json',
            str(self.video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            return data.get('frames', [])
        except:
            return []
    
    def _select_frames(self, frame_types: List[Dict], num_frames: int, 
                      max_frames: int) -> List[int]:
        """Select frames to visualize (I-frame followed by P/B frames)."""
        indices = []
        
        # Find first I-frame
        for i, ft in enumerate(frame_types[:50]):
            if ft.get('pict_type') == 'I':
                indices.append(i)
                break
        
        # Add subsequent frames
        if indices:
            start = indices[0]
            for i in range(start + 1, min(start + num_frames, len(frame_types), max_frames)):
                indices.append(i)
        
        return indices
    
    def _read_frames(self, frame_indices: List[int], frame_types: List[Dict],
                    mv_data: np.ndarray, block_size: int) -> List[Dict]:
        """Read frames and prepare visualization data."""
        frames_data = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        all_frames = []
        
        # Read all needed frames
        while len(all_frames) <= max(frame_indices) + 1:
            ret, frame = self.cap.read()
            if not ret:
                break
            all_frames.append(frame)
        
        # Process selected frames
        for idx in frame_indices:
            if idx < len(all_frames) and idx < len(mv_data):
                frames_data.append({
                    'frame': all_frames[idx],
                    'prev_frame': all_frames[idx-1] if idx > 0 else all_frames[idx],
                    'type': frame_types[idx].get('pict_type', '?') if idx < len(frame_types) else '?',
                    'index': idx,
                    'motion_vectors': mv_data[idx]
                })
        
        return frames_data
    
    def _visualize_motion_blocks(self, motion_vectors: np.ndarray, 
                                block_size: int) -> np.ndarray:
        """Visualize motion vectors as colored blocks."""
        vis = np.ones((self.height, self.width, 3), dtype=np.uint8) * 230
        grid_h, grid_w = motion_vectors.shape[:2]
        
        # Ensure we don't exceed image bounds
        max_grid_h = (self.height + block_size - 1) // block_size
        max_grid_w = (self.width + block_size - 1) // block_size
        grid_h = min(grid_h, max_grid_h)
        grid_w = min(grid_w, max_grid_w)
        
        for i in range(grid_h):
            for j in range(grid_w):
                y = i * block_size
                x = j * block_size
                
                # Calculate actual block dimensions (handle edge blocks)
                block_h = min(block_size, self.height - y)
                block_w = min(block_size, self.width - x)
                
                # Skip if block is completely outside image bounds
                if block_h <= 0 or block_w <= 0 or y >= self.height or x >= self.width:
                    continue
                
                # Get motion vector magnitude
                mv_x = motion_vectors[i, j, 0]
                mv_y = motion_vectors[i, j, 1]
                magnitude = np.sqrt(mv_x**2 + mv_y**2)
                
                # Color based on motion
                if magnitude < 0.5:
                    color = (240, 220, 200)  # Light blue/gray
                else:
                    # HSV color mapping
                    angle = np.arctan2(mv_y, mv_x) + np.pi
                    hue = int(angle / (2 * np.pi) * 180)
                    saturation = min(255, int(magnitude * 20))
                    
                    hsv_color = np.array([[[hue, saturation, 255]]], dtype=np.uint8)
                    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
                    color = tuple(int(c) for c in bgr_color)
                
                # Draw block (ensuring we stay within image bounds)
                x1, y1 = x, y
                x2 = min(x + block_w - 1, self.width - 1)
                y2 = min(y + block_h - 1, self.height - 1)
                
                # Draw filled block
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, -1)
                
                # Draw grid lines (skip for last row and column to avoid edge artifacts)
                draw_right_line = (j < grid_w - 1) and (x2 < self.width - 1)
                draw_bottom_line = (i < grid_h - 1) and (y2 < self.height - 1)
                
                if draw_right_line:
                    cv2.line(vis, (x2, y1), (x2, y2), (200, 200, 200), 1)
                if draw_bottom_line:
                    cv2.line(vis, (x1, y2), (x2, y2), (200, 200, 200), 1)
        
        return vis
    
    def _compute_residuals(self, curr_frame: np.ndarray, prev_frame: np.ndarray,
                          motion_vectors: np.ndarray, block_size: int) -> np.ndarray:
        """Compute residuals using motion compensation."""
        compensated = np.zeros_like(curr_frame, dtype=np.uint8)
        grid_h, grid_w = motion_vectors.shape[:2]
        
        # Apply motion compensation
        for i in range(grid_h):
            for j in range(grid_w):
                dst_y = i * block_size
                dst_x = j * block_size
                
                # Calculate actual block dimensions (handle edge blocks)
                block_h = min(block_size, self.height - dst_y)
                block_w = min(block_size, self.width - dst_x)
                
                # Skip if block is completely outside image bounds
                if block_h <= 0 or block_w <= 0:
                    continue
                
                # Motion vectors point from reference to current
                mv_x = -motion_vectors[i, j, 0]
                mv_y = -motion_vectors[i, j, 1]
                
                src_y = int(round(dst_y + mv_y))
                src_x = int(round(dst_x + mv_x))
                
                # Ensure source block is within bounds
                src_y = max(0, min(src_y, self.height - block_h))
                src_x = max(0, min(src_x, self.width - block_w))
                
                # Copy the block from source to destination
                compensated[dst_y:dst_y+block_h, dst_x:dst_x+block_w] = \
                    prev_frame[src_y:src_y+block_h, src_x:src_x+block_w]
        
        # Compute residuals
        diff = cv2.absdiff(curr_frame, compensated)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Create visualization
        vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Find significant differences
        _, thresh = cv2.threshold(diff_gray, 15, 255, cv2.THRESH_BINARY)
        
        # Clean up noise
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find and draw contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:
                # Get intensity in this region
                mask = np.zeros(diff_gray.shape, np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_val = cv2.mean(diff_gray, mask=mask)[0]
                
                # Color based on intensity
                if mean_val > 40:
                    color = (0, 165, 255)  # Orange
                elif mean_val > 20:
                    color = (0, 200, 255)  # Light orange
                else:
                    color = (200, 200, 100)  # Light yellow
                
                # Draw filled contour
                overlay = vis.copy()
                cv2.drawContours(overlay, [contour], -1, color, -1)
                cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
                cv2.drawContours(vis, [contour], -1, color, 1)
        
        return vis
    
    def _create_grid_visualization(self, frames_data: List[Dict], block_size: int,
                                  output_path: Optional[str] = None) -> str:
        """Create the grid visualization."""
        n_frames = len(frames_data)
        
        # Calculate figure size based on actual frame dimensions
        # Each frame should be displayed at its actual size in inches (assuming 100 DPI)
        dpi = 100
        frame_width_inches = self.width / dpi
        frame_height_inches = self.height / dpi
        
        # Total figure size: label column + n_frames columns, 3 rows
        label_width = 0.5  # inches for label column
        total_width = label_width + n_frames * frame_width_inches + 0.5  # add some padding
        total_height = 3 * frame_height_inches + 0.5  # 3 rows plus padding
        
        fig = plt.figure(figsize=(total_width, total_height), dpi=dpi)
        
        # Create grid layout
        gs = GridSpec(3, n_frames + 1, figure=fig,
                      width_ratios=[0.5] + [1]*n_frames,
                      height_ratios=[1, 1, 1],
                      hspace=0.05, wspace=0.05)
        
        # Row labels
        labels = ['Original', 'Motion\nvectors', 'Residuals']
        for i, label in enumerate(labels):
            ax = fig.add_subplot(gs[i, 0])
            ax.text(0.5, 0.5, label, rotation=90, ha='center', va='center',
                   fontsize=16, weight='bold')
            ax.axis('off')
        
        # Process each frame
        for i, data in enumerate(frames_data):
            frame = data['frame']
            prev_frame = data['prev_frame']
            frame_type = data['type']
            motion_vectors = data['motion_vectors']
            
            # Original frame
            ax1 = fig.add_subplot(gs[0, i+1])
            ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax1.set_title(f'{frame_type}-frame', fontsize=20, weight='bold', pad=10)
            ax1.axis('off')
            
            # Motion vectors
            ax2 = fig.add_subplot(gs[1, i+1])
            if frame_type != 'I':
                mv_vis = self._visualize_motion_blocks(motion_vectors, block_size)
                ax2.imshow(cv2.cvtColor(mv_vis, cv2.COLOR_BGR2RGB))
            else:
                blank = np.ones((self.height, self.width, 3), dtype=np.uint8)
                blank[:, :] = [240, 240, 200]
                ax2.imshow(blank)
            ax2.axis('off')
            
            # Residuals
            ax3 = fig.add_subplot(gs[2, i+1])
            if frame_type != 'I':
                residuals = self._compute_residuals(frame, prev_frame, motion_vectors, block_size)
                ax3.imshow(cv2.cvtColor(residuals, cv2.COLOR_BGR2RGB))
            else:
                # I-frame visualization
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 30, 100)
                vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                vis[:, :, 0] = edges * 0.3
                vis[:, :, 1] = edges * 0.7
                vis[:, :, 2] = edges * 1.0
                vis[:, :, 1] += (gray // 8)
                vis[:, :, 2] += (gray // 8)
                ax3.imshow(vis)
            ax3.axis('off')
        
        # Add "..." if more frames exist
        if n_frames < 10:  # Arbitrary threshold
            ax_dots = fig.add_subplot(gs[:, -1])
            ax_dots.text(1.1, 0.5, '...', transform=ax_dots.transAxes,
                        fontsize=30, ha='left', va='center')
            ax_dots.axis('off')
        
        plt.tight_layout()
        
        # Save
        if output_path is None:
            output_path = f'outputs/visualizations/motion_residuals_{block_size}x{block_size}.png'
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Saved visualization to {output_path}")
        return output_path


def visualize_motion_residuals(video_path: str, motion_vectors_path: str,
                              output_path: Optional[str] = None,
                              num_frames: int = 5) -> str:
    """
    Create motion vector and residual visualization.
    
    Args:
        video_path: Path to video file
        motion_vectors_path: Path to motion vectors .npy file
        output_path: Output path for visualization
        num_frames: Number of frames to visualize
        
    Returns:
        Path to saved visualization
    """
    visualizer = MotionResidualVisualizer(video_path)
    return visualizer.create_visualization(motion_vectors_path, output_path, num_frames)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m visualizers.residual_grid <video_path> <motion_vectors_path> [output_path]")
        print("Example: python -m visualizers.residual_grid video.mp4 motion_vectors_4x4.npy")
        sys.exit(1)
    
    video_path = sys.argv[1]
    mv_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    visualize_motion_residuals(video_path, mv_path, output_path)
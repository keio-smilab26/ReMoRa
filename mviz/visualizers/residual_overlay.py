#!/usr/bin/env python3
"""
Fixed visualization of frames with actual codec motion vectors and residuals.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import json
import logging
from typing import List, Dict, Tuple
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodecMotionVisualizerFixed:
    """Fixed visualizer for codec motion vectors."""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {self.width}x{self.height}, {self.frame_count} frames")
    
    def load_codec_motion_vectors(self, mv_file: str) -> np.ndarray:
        """Load codec motion vectors."""
        mv_data = np.load(mv_file)
        logger.info(f"Loaded codec motion vectors: {mv_data.shape}")
        return mv_data
    
    def get_frame_types(self) -> List[Dict]:
        """Get frame type information."""
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
    
    def visualize_motion_blocks_codec(self, motion_vectors: np.ndarray, block_size: int) -> np.ndarray:
        """Visualize codec motion vectors as colored blocks."""
        # Create visualization at video resolution
        vis = np.ones((self.height, self.width, 3), dtype=np.uint8) * 230
        
        grid_h, grid_w = motion_vectors.shape[:2]
        
        # Draw blocks with colors based on motion
        for i in range(grid_h):
            for j in range(grid_w):
                # Block position in image
                y = i * block_size
                x = j * block_size
                
                # Skip if out of bounds
                if y + block_size > self.height or x + block_size > self.width:
                    continue
                
                # Get motion vector
                mv_x = motion_vectors[i, j, 0]
                mv_y = motion_vectors[i, j, 1]
                magnitude = np.sqrt(mv_x**2 + mv_y**2)
                
                # Color based on motion
                if magnitude < 0.5:
                    # No/minimal motion - light blue
                    color = (240, 220, 200)
                else:
                    # Motion present - use HSV color mapping
                    angle = np.arctan2(mv_y, mv_x) + np.pi
                    hue = int(angle / (2 * np.pi) * 180)
                    saturation = min(255, int(magnitude * 20))
                    value = 255
                    
                    hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
                    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
                    color = tuple(int(c) for c in bgr_color)
                
                # Draw filled block
                cv2.rectangle(vis, (x, y), 
                             (min(x + block_size - 1, self.width-1), 
                              min(y + block_size - 1, self.height-1)),
                             color, -1)
                
                # Add subtle grid lines
                cv2.rectangle(vis, (x, y), 
                             (min(x + block_size - 1, self.width-1), 
                              min(y + block_size - 1, self.height-1)),
                             (200, 200, 200), 1)
        
        return vis
    
    def compute_residuals_with_codec_mvs(self, curr_frame: np.ndarray, prev_frame: np.ndarray,
                                        motion_vectors: np.ndarray, block_size: int) -> np.ndarray:
        """Compute residuals using actual codec motion vectors - FIXED version."""
        # Create motion compensated frame
        compensated = np.zeros_like(curr_frame, dtype=np.uint8)
        
        grid_h, grid_w = motion_vectors.shape[:2]
        
        # First, check if there's any significant motion
        total_motion = np.sum(np.abs(motion_vectors))
        
        if total_motion < 1.0:  # Very little motion
            # For static scenes, residuals should be minimal
            compensated = prev_frame.copy()
        else:
            # Apply motion compensation block by block
            for i in range(grid_h):
                for j in range(grid_w):
                    # Current block position (destination)
                    dst_y = i * block_size
                    dst_x = j * block_size
                    
                    # Skip if out of bounds
                    if dst_y + block_size > self.height or dst_x + block_size > self.width:
                        continue
                    
                    # Get motion vector (points from current to reference)
                    # Negative because MV points from reference to current
                    mv_x = -motion_vectors[i, j, 0]
                    mv_y = -motion_vectors[i, j, 1]
                    
                    # Source position in previous frame
                    src_y = int(round(dst_y + mv_y))
                    src_x = int(round(dst_x + mv_x))
                    
                    # Extract blocks with bounds checking
                    block_h = min(block_size, self.height - dst_y)
                    block_w = min(block_size, self.width - dst_x)
                    
                    if (0 <= src_y and src_y + block_h <= self.height and
                        0 <= src_x and src_x + block_w <= self.width):
                        # Copy from source in prev frame to destination in compensated
                        compensated[dst_y:dst_y+block_h, dst_x:dst_x+block_w] = \
                            prev_frame[src_y:src_y+block_h, src_x:src_x+block_w]
                    else:
                        # Use same position if out of bounds
                        compensated[dst_y:dst_y+block_h, dst_x:dst_x+block_w] = \
                            prev_frame[dst_y:dst_y+block_h, dst_x:dst_x+block_w]
        
        # Compute residuals
        diff = cv2.absdiff(curr_frame, compensated)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Create visualization on black background
        vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Method 1: Show significant differences as contours
        _, thresh = cv2.threshold(diff_gray, 15, 255, cv2.THRESH_BINARY)
        
        # Clean up noise
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours and filled regions
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:  # Skip tiny noise
                # Get average intensity in this region
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
                
                # Draw filled contour with transparency
                overlay = vis.copy()
                cv2.drawContours(overlay, [contour], -1, color, -1)
                cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
                
                # Draw contour outline
                cv2.drawContours(vis, [contour], -1, color, 1)
        
        return vis
    
    def create_codec_visualization(self, num_frames: int = 5, block_size: int = 16):
        """Create visualization using actual codec motion vectors."""
        
        # Load codec motion vectors
        mv_file = f'codec_motion_vectors/codec_mv_{block_size}x{block_size}.npy'
        if not Path(mv_file).exists():
            logger.error(f"Motion vector file not found: {mv_file}")
            return
        
        codec_mvs = self.load_codec_motion_vectors(mv_file)
        
        # Get frame types
        frame_types = self.get_frame_types()
        
        # Select frames to visualize
        frame_indices = []
        
        # Find I-frame
        for i, ft in enumerate(frame_types[:50]):
            if ft.get('pict_type') == 'I':
                frame_indices.append(i)
                break
        
        # Add subsequent frames
        if frame_indices:
            start = frame_indices[0]
            for i in range(start + 1, min(start + num_frames, len(frame_types), len(codec_mvs))):
                frame_indices.append(i)
        
        # Read frames
        frames_data = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        all_frames = []
        
        while len(all_frames) <= max(frame_indices) + 1:
            ret, frame = self.cap.read()
            if not ret:
                break
            all_frames.append(frame)
        
        self.cap.release()
        
        # Process selected frames
        for idx in frame_indices:
            if idx < len(all_frames) and idx < len(codec_mvs):
                frames_data.append({
                    'frame': all_frames[idx],
                    'prev_frame': all_frames[idx-1] if idx > 0 else all_frames[idx],
                    'type': frame_types[idx].get('pict_type', '?') if idx < len(frame_types) else '?',
                    'index': idx,
                    'motion_vectors': codec_mvs[idx]
                })
        
        # Create figure
        n_frames = len(frames_data)
        fig = plt.figure(figsize=(4*n_frames + 1, 9))
        
        # Create grid
        gs = GridSpec(3, n_frames + 1, figure=fig,
                      width_ratios=[0.5] + [1]*n_frames,
                      height_ratios=[1, 1, 1],
                      hspace=0.05, wspace=0.05)
        
        # Row labels
        ax_label1 = fig.add_subplot(gs[0, 0])
        ax_label1.text(0.5, 0.5, 'Original', rotation=90, ha='center', va='center',
                      fontsize=16, weight='bold')
        ax_label1.axis('off')
        
        ax_label2 = fig.add_subplot(gs[1, 0])
        ax_label2.text(0.5, 0.5, 'Motion\nvectors', rotation=90, ha='center', va='center',
                      fontsize=16, weight='bold')
        ax_label2.axis('off')
        
        ax_label3 = fig.add_subplot(gs[2, 0])
        ax_label3.text(0.5, 0.5, 'Residuals', rotation=90, ha='center', va='center',
                      fontsize=16, weight='bold')
        ax_label3.axis('off')
        
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
                mv_vis = self.visualize_motion_blocks_codec(motion_vectors, block_size)
                ax2.imshow(cv2.cvtColor(mv_vis, cv2.COLOR_BGR2RGB))
            else:
                # Solid light color for I-frame
                blank = np.ones((self.height, self.width, 3), dtype=np.uint8)
                blank[:, :] = [240, 240, 200]
                ax2.imshow(blank)
            ax2.axis('off')
            
            # Residuals
            ax3 = fig.add_subplot(gs[2, i+1])
            if frame_type != 'I':
                residuals = self.compute_residuals_with_codec_mvs(
                    frame, prev_frame, motion_vectors, block_size)
                ax3.imshow(cv2.cvtColor(residuals, cv2.COLOR_BGR2RGB))
            else:
                # For I-frame, show the whole frame content in a stylized way
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Create edge-based representation
                edges = cv2.Canny(gray, 30, 100)
                
                # Create colored visualization
                vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                vis[:, :, 0] = edges * 0.3  # Blue channel
                vis[:, :, 1] = edges * 0.7  # Green channel  
                vis[:, :, 2] = edges * 1.0  # Red channel
                
                # Add some of the original intensity
                vis[:, :, 1] += (gray // 8)
                vis[:, :, 2] += (gray // 8)
                
                ax3.imshow(vis)
            ax3.axis('off')
        
        # Add "..." at the end
        if n_frames < len(frame_types):
            ax_dots = fig.add_subplot(gs[:, -1])
            ax_dots.text(1.1, 0.5, '...', transform=ax_dots.transAxes,
                        fontsize=30, ha='left', va='center')
            ax_dots.axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = f'codec_motion_residuals_visualization_fixed_{block_size}x{block_size}.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Saved visualization to {output_path}")
        
        return output_path


def main():
    """Create fixed visualization with actual codec motion vectors."""
    import argparse
    parser = argparse.ArgumentParser(description='Visualize codec motion vectors and residuals.')
    parser.add_argument('video', type=str, help='Path to the input video file')
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    logger.info("=== CREATING FIXED CODEC MOTION VECTOR VISUALIZATION ===")
    
    # Create visualizer
    visualizer = CodecMotionVisualizerFixed(video_path)
    
    # Create visualization with 4x4 blocks for finer detail
    output_path = visualizer.create_codec_visualization(num_frames=5, block_size=4)
    
    print("\n=== VISUALIZATION COMPLETE ===")
    print(f"Created: {output_path}")
    print("\nThis visualization shows:")
    print("- Original frames (I, P, B)")
    print("- ACTUAL codec motion vectors as colored blocks")
    print("- Fixed residuals after motion compensation")
    print("\nThe residuals now correctly show only the differences!")


if __name__ == "__main__":
    main()
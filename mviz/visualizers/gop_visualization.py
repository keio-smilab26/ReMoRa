#!/usr/bin/env python3
"""
GOP-aware motion vector visualization showing all frames in each GOP.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_gop_motion_visualization(
    video_path: str, 
    motion_vectors: np.ndarray,
    output_dir: str, 
    gop_size: int = 8,
    block_size: int = 4,
    arrow_scale: float = 1.0,
    max_gops: int = 5
):
    """
    Create motion vector visualizations for each GOP showing all frames.
    
    Args:
        video_path: Path to video file
        motion_vectors: Motion vectors array (T, H, W, 2)
        output_dir: Output directory for visualizations
        gop_size: Number of frames per GOP
        block_size: Block size used for motion vectors
        arrow_scale: Scale factor for arrows
        max_gops: Maximum number of GOPs to visualize
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    video_name = Path(video_path).stem
    
    total_frames = motion_vectors.shape[0]
    num_gops = min(total_frames // gop_size, max_gops)
    
    logger.info(f"Creating GOP visualizations for {num_gops} GOPs (GOP size: {gop_size})")
    
    for gop_idx in range(num_gops):
        start_frame = gop_idx * gop_size
        end_frame = min(start_frame + gop_size, total_frames)
        frames_in_gop = end_frame - start_frame
        
        # Read frames for this GOP
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for i in range(frames_in_gop):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break
        
        if not frames:
            continue
            
        # Create visualization for this GOP
        fig = plt.figure(figsize=(20, 12))
        
        # Calculate grid layout (2 rows per GOP: original frames + motion vectors)
        cols = frames_in_gop
        
        # Create subplots
        for i, frame in enumerate(frames):
            frame_idx = start_frame + i
            
            # Top row: Original frame
            ax1 = plt.subplot(3, cols, i + 1)
            ax1.imshow(frame)
            
            # Mark I-frame vs P/B-frame
            frame_type = "I-frame" if i == 0 else "P/B-frame"
            ax1.set_title(f'Frame {frame_idx}\n{frame_type}', fontsize=10)
            ax1.axis('off')
            
            # Middle row: Motion vectors on frame
            ax2 = plt.subplot(3, cols, cols + i + 1)
            ax2.imshow(frame, alpha=0.3)
            
            # Get motion vectors for this frame
            mv_frame = motion_vectors[frame_idx]
            h, w = frame.shape[:2]
            mv_h, mv_w = mv_frame.shape[:2]
            
            # Create grid for quiver plot
            Y, X = np.mgrid[0:mv_h, 0:mv_w]
            U = mv_frame[:, :, 0]
            V = mv_frame[:, :, 1]
            
            # Scale coordinates to image size
            X_scaled = X * (w / mv_w) + block_size / 2
            Y_scaled = Y * (h / mv_h) + block_size / 2
            
            # Calculate magnitude for coloring
            magnitude = np.sqrt(U**2 + V**2)
            
            # Skip if no motion (I-frame)
            if magnitude.max() > 0:
                # Filter out zero motion vectors for cleaner visualization
                mask = magnitude > 0.1
                
                if mask.any():
                    q = ax2.quiver(
                        X_scaled[mask], Y_scaled[mask], 
                        U[mask], V[mask],
                        magnitude[mask],
                        scale=50 / arrow_scale,
                        scale_units='width',
                        width=0.003,
                        cmap='hot',
                        alpha=0.9
                    )
            
            ax2.set_title(f'Motion Vectors\nMax: {magnitude.max():.2f}', fontsize=10)
            ax2.axis('off')
            
            # Bottom row: Motion magnitude heatmap
            ax3 = plt.subplot(3, cols, 2*cols + i + 1)
            im = ax3.imshow(magnitude, cmap='hot', interpolation='bilinear', vmin=0, vmax=5)
            ax3.set_title(f'Motion Magnitude\nMean: {magnitude.mean():.2f}', fontsize=10)
            ax3.axis('off')
        
        # Add GOP title
        fig.suptitle(f'{video_name} - GOP {gop_idx + 1} (Frames {start_frame}-{end_frame-1})', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar for magnitude
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
        plt.colorbar(im, cax=cbar_ax, label='Motion Magnitude (pixels)')
        
        plt.tight_layout()
        
        # Save GOP visualization
        output_path = output_dir / f"{video_name}_GOP_{gop_idx+1:03d}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved GOP {gop_idx + 1} visualization: {output_path}")
    
    cap.release()
    
    # Create a summary visualization showing motion intensity per GOP
    create_gop_summary(motion_vectors, output_dir / f"{video_name}_GOP_summary.png", 
                      gop_size, num_gops)


def create_gop_summary(motion_vectors: np.ndarray, output_path: str, 
                       gop_size: int, num_gops: int):
    """
    Create a summary showing motion intensity patterns across GOPs.
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Calculate motion magnitude
    magnitude = np.sqrt(motion_vectors[..., 0]**2 + motion_vectors[..., 1]**2)
    
    # Per-frame statistics
    mean_motion = magnitude.mean(axis=(1, 2))
    max_motion = magnitude.max(axis=(1, 2))
    
    frames = np.arange(len(mean_motion))
    
    # Plot 1: Motion over time with GOP boundaries
    ax1 = axes[0]
    ax1.plot(frames, mean_motion, 'b-', label='Mean Motion', linewidth=1.5)
    ax1.plot(frames, max_motion, 'r--', label='Max Motion', alpha=0.7)
    
    # Add GOP boundaries
    for i in range(num_gops + 1):
        x = i * gop_size
        if x < len(frames):
            ax1.axvline(x, color='green', linestyle=':', alpha=0.5)
            if i < num_gops:
                ax1.text(x + gop_size/2, ax1.get_ylim()[1] * 0.95, 
                        f'GOP {i+1}', ha='center', fontsize=9)
    
    # Mark I-frames
    i_frame_positions = [i * gop_size for i in range(num_gops) if i * gop_size < len(frames)]
    i_frame_motions = [mean_motion[pos] if pos < len(mean_motion) else 0 
                       for pos in i_frame_positions]
    ax1.scatter(i_frame_positions, i_frame_motions, 
               color='red', s=50, zorder=5, label='I-frames')
    
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Motion Magnitude (pixels)')
    ax1.set_title('Motion Intensity Across GOPs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Per-GOP statistics
    ax2 = axes[1]
    gop_means = []
    gop_maxs = []
    gop_stds = []
    
    for i in range(num_gops):
        start = i * gop_size
        end = min(start + gop_size, len(mean_motion))
        gop_data = magnitude[start:end]
        
        gop_means.append(gop_data.mean())
        gop_maxs.append(gop_data.max())
        gop_stds.append(gop_data.std())
    
    x_pos = np.arange(num_gops)
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, gop_means, width, label='Mean', color='blue', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, gop_maxs, width, label='Max', color='red', alpha=0.7)
    
    # Add error bars for std
    ax2.errorbar(x_pos - width/2, gop_means, yerr=gop_stds, 
                fmt='none', color='black', alpha=0.5, capsize=3)
    
    ax2.set_xlabel('GOP Number')
    ax2.set_ylabel('Motion Magnitude (pixels)')
    ax2.set_title('Motion Statistics per GOP')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'GOP {i+1}' for i in range(num_gops)])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved GOP summary: {output_path}")


def create_motion_overlay_video(video_path: str, output_path: str):
    """
    Create a video with FFmpeg codec motion vectors overlaid on frames.

    Uses FFmpeg's codecview filter to render the raw MV arrows directly.

    Args:
        video_path: Path to source video file
        output_path: Output path for the overlay video
    """
    import subprocess

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg', '-y',
        '-flags2', '+export_mvs',
        '-i', video_path,
        '-vf', 'codecview=mv=pf+bf+bb',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    logger.info(f"Saved motion overlay video: {output_path}")
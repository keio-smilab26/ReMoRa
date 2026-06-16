#!/usr/bin/env python3
"""
Utility functions for creating motion vector visualizations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional


def create_motion_visualization(video_path: str, motion_vectors: np.ndarray,
                              output_path: str, block_size: int = 8,
                              num_frames: int = 5, arrow_scale: float = 1.0):
    """
    Create a simple motion vector visualization.
    
    Args:
        video_path: Path to video file
        motion_vectors: Motion vectors array (T, H, W, 2)
        output_path: Output path for visualization
        block_size: Block size used
        num_frames: Number of frames to visualize
        arrow_scale: Scale factor for arrows
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get frames
    frames = []
    for i in range(min(num_frames, motion_vectors.shape[0])):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    
    if not frames:
        return
    
    # Create figure
    fig, axes = plt.subplots(2, num_frames, figsize=(15, 6))
    if num_frames == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(len(frames)):
        # Original frame
        axes[0, i].imshow(frames[i])
        axes[0, i].set_title(f'Frame {i}')
        axes[0, i].axis('off')
        
        # Motion vectors
        h, w = frames[i].shape[:2]
        mv_h, mv_w = motion_vectors.shape[1:3]
        
        # Create quiver plot
        Y, X = np.mgrid[0:mv_h, 0:mv_w]
        U = motion_vectors[i, :, :, 0]
        V = motion_vectors[i, :, :, 1]
        
        # Scale coordinates to image size
        X_scaled = X * (w / mv_w) + block_size / 2
        Y_scaled = Y * (h / mv_h) + block_size / 2
        
        # Plot on frame
        axes[1, i].imshow(frames[i], alpha=0.5)
        
        # Magnitude for coloring
        magnitude = np.sqrt(U**2 + V**2)
        
        # Quiver plot
        q = axes[1, i].quiver(
            X_scaled, Y_scaled, U, V,
            magnitude,
            scale=50 / arrow_scale,
            scale_units='width',
            cmap='hot',
            alpha=0.8
        )
        
        axes[1, i].set_title(f'Motion Vectors')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_motion_heatmap(motion_vectors: np.ndarray, output_path: str,
                         aggregate: str = 'magnitude'):
    """
    Create a heatmap of motion vector magnitudes or directions.
    
    Args:
        motion_vectors: Motion vectors array (T, H, W, 2)
        output_path: Output path
        aggregate: 'magnitude', 'direction', or 'both'
    """
    # Calculate magnitude
    magnitude = np.sqrt(
        motion_vectors[..., 0]**2 + motion_vectors[..., 1]**2
    )
    
    # Average over time
    avg_magnitude = magnitude.mean(axis=0)
    
    if aggregate == 'magnitude':
        plt.figure(figsize=(10, 8))
        plt.imshow(avg_magnitude, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Average Motion Magnitude')
        plt.title('Motion Magnitude Heatmap')
        
    elif aggregate == 'direction':
        # Calculate average direction
        avg_u = motion_vectors[..., 0].mean(axis=0)
        avg_v = motion_vectors[..., 1].mean(axis=0)
        direction = np.arctan2(avg_v, avg_u)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(direction, cmap='hsv', interpolation='nearest')
        plt.colorbar(label='Motion Direction (radians)')
        plt.title('Motion Direction Heatmap')
        
    elif aggregate == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Magnitude
        im1 = ax1.imshow(avg_magnitude, cmap='hot', interpolation='nearest')
        plt.colorbar(im1, ax=ax1, label='Average Motion Magnitude')
        ax1.set_title('Motion Magnitude')
        
        # Direction
        avg_u = motion_vectors[..., 0].mean(axis=0)
        avg_v = motion_vectors[..., 1].mean(axis=0)
        direction = np.arctan2(avg_v, avg_u)
        
        im2 = ax2.imshow(direction, cmap='hsv', interpolation='nearest')
        plt.colorbar(im2, ax=ax2, label='Motion Direction (radians)')
        ax2.set_title('Motion Direction')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_temporal_motion_profile(motion_vectors: np.ndarray, output_path: str):
    """
    Create a temporal profile of motion over time.
    
    Args:
        motion_vectors: Motion vectors array (T, H, W, 2)
        output_path: Output path
    """
    # Calculate per-frame statistics
    magnitude = np.sqrt(
        motion_vectors[..., 0]**2 + motion_vectors[..., 1]**2
    )
    
    mean_motion = magnitude.mean(axis=(1, 2))
    std_motion = magnitude.std(axis=(1, 2))
    max_motion = magnitude.max(axis=(1, 2))
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    frames = np.arange(len(mean_motion))
    
    # Mean motion with std
    ax1.plot(frames, mean_motion, 'b-', label='Mean')
    ax1.fill_between(
        frames,
        mean_motion - std_motion,
        mean_motion + std_motion,
        alpha=0.3,
        label='±1 STD'
    )
    ax1.plot(frames, max_motion, 'r--', alpha=0.7, label='Max')
    ax1.set_ylabel('Motion Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Motion Magnitude Over Time')
    
    # Motion distribution
    percentiles = [10, 25, 50, 75, 90]
    colors = plt.cm.viridis(np.linspace(0, 1, len(percentiles)))
    
    for p, color in zip(percentiles, colors):
        values = np.percentile(magnitude, p, axis=(1, 2))
        ax2.plot(frames, values, color=color, label=f'{p}th percentile')
    
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Motion Magnitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Motion Distribution Percentiles')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
#!/usr/bin/env python3
"""
Example of how to use extracted motion vectors in a deep learning model.
"""

import numpy as np
import json
from pathlib import Path


def load_motion_vectors(base_path: str):
    """Load motion vectors and metadata."""
    base_path = Path(base_path)
    
    # Load motion vectors
    mv_data = np.load(base_path.with_suffix('.npy'))
    
    # Load metadata
    with open(base_path.with_suffix('.json'), 'r') as f:
        metadata = json.load(f)
    
    return mv_data, metadata


def prepare_motion_features_for_ml(motion_vectors: np.ndarray):
    """
    Prepare motion vector features for machine learning.
    
    Input shape: (n_frames, h//mb_size, w//mb_size, 2)
    Output: Various feature representations
    """
    n_frames, h_blocks, w_blocks, _ = motion_vectors.shape
    
    # 1. Motion magnitude (speed)
    magnitude = np.sqrt(motion_vectors[..., 0]**2 + motion_vectors[..., 1]**2)
    print(f"Motion magnitude shape: {magnitude.shape}")
    
    # 2. Motion direction (angle)
    direction = np.arctan2(motion_vectors[..., 1], motion_vectors[..., 0])
    print(f"Motion direction shape: {direction.shape}")
    
    # 3. Temporal motion difference
    motion_diff = np.diff(motion_vectors, axis=0)
    print(f"Motion difference shape: {motion_diff.shape}")
    
    # 4. Global motion statistics per frame
    frame_stats = {
        'mean_motion': np.mean(magnitude, axis=(1, 2)),
        'max_motion': np.max(magnitude, axis=(1, 2)),
        'motion_variance': np.var(magnitude, axis=(1, 2))
    }
    
    # 5. Spatial motion patterns (e.g., for each frame, dominant motion direction)
    dominant_direction = np.zeros((n_frames, 8))  # 8 directional bins
    for i in range(n_frames):
        # Quantize directions into 8 bins
        dir_bins = ((direction[i] + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        for j in range(8):
            dominant_direction[i, j] = np.sum(dir_bins == j)
    
    return {
        'magnitude': magnitude,
        'direction': direction,
        'motion_diff': motion_diff,
        'frame_stats': frame_stats,
        'dominant_direction': dominant_direction
    }


def example_pytorch_usage():
    """Example of using motion vectors in PyTorch."""
    try:
        import torch
        import torch.nn as nn
        
        # Load motion vectors
        mv_data, metadata = load_motion_vectors('motion_vectors_ml/motion_vectors_4x4')
        print(f"Loaded motion vectors: {mv_data.shape}")
        
        # Convert to PyTorch tensor
        mv_tensor = torch.from_numpy(mv_data).float()
        
        # Example: Simple CNN for motion pattern recognition
        class MotionFeatureExtractor(nn.Module):
            def __init__(self, input_channels=2):
                super().__init__()
                self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2)
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                
            def forward(self, x):
                # x shape: (batch, 2, height, width)
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                x = torch.relu(self.conv2(x))
                x = self.global_pool(x)
                return x.squeeze(-1).squeeze(-1)
        
        # Create model
        model = MotionFeatureExtractor()
        
        # Process motion vectors frame by frame
        batch_size = 8
        features = []
        
        for i in range(0, len(mv_tensor), batch_size):
            batch = mv_tensor[i:i+batch_size]
            # Reshape: (batch, h, w, 2) -> (batch, 2, h, w)
            batch = batch.permute(0, 3, 1, 2)
            
            with torch.no_grad():
                feat = model(batch)
                features.append(feat)
        
        features = torch.cat(features, dim=0)
        print(f"Extracted features shape: {features.shape}")
        
    except ImportError:
        print("PyTorch not installed. Showing numpy example only.")


def example_tensorflow_usage():
    """Example of using motion vectors in TensorFlow."""
    try:
        import tensorflow as tf
        
        # Load motion vectors
        mv_data, metadata = load_motion_vectors('motion_vectors_ml/motion_vectors_4x4')
        print(f"Loaded motion vectors: {mv_data.shape}")
        
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(90, 160, 2)),
            tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10)  # Example: 10 action classes
        ])
        
        # Process motion vectors
        predictions = model.predict(mv_data, batch_size=8)
        print(f"Predictions shape: {predictions.shape}")
        
    except ImportError:
        print("TensorFlow not installed. Showing numpy example only.")


def main():
    """Main function demonstrating motion vector usage."""
    
    # Load motion vectors
    mv_data, metadata = load_motion_vectors('motion_vectors_ml/motion_vectors_4x4')
    
    print("=== MOTION VECTOR DATA ===")
    print(f"Shape: {metadata['shape']}")
    print(f"Data type: {metadata['dtype']}")
    print(f"Macroblock size: {metadata['macroblock_size']}x{metadata['macroblock_size']}")
    print(f"Description: {metadata['description']}")
    
    # Prepare features
    print("\n=== PREPARING FEATURES ===")
    features = prepare_motion_features_for_ml(mv_data)
    
    print(f"\nFrame statistics shape: {features['frame_stats']['mean_motion'].shape}")
    print(f"Mean motion per frame (first 10): {features['frame_stats']['mean_motion'][:10]}")
    
    # Show how to use in deep learning
    print("\n=== DEEP LEARNING USAGE ===")
    print("\nFor PyTorch:")
    print("```python")
    print("import torch")
    print("mv_tensor = torch.from_numpy(mv_data).float()")
    print("# Shape: (n_frames, h_blocks, w_blocks, 2)")
    print("# Reshape for CNN: (n_frames, 2, h_blocks, w_blocks)")
    print("mv_tensor = mv_tensor.permute(0, 3, 1, 2)")
    print("```")
    
    print("\nFor TensorFlow:")
    print("```python")
    print("import tensorflow as tf")
    print("mv_tensor = tf.constant(mv_data)")
    print("# Shape: (n_frames, h_blocks, w_blocks, 2)")
    print("# Already in correct format for Conv2D")
    print("```")
    
    # Try to run framework examples
    print("\n=== FRAMEWORK EXAMPLES ===")
    example_pytorch_usage()
    example_tensorflow_usage()


if __name__ == "__main__":
    main()
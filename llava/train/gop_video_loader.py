"""
GOP-aware video loader that loads I-frames as images and P/B-frames as motion vectors.
Preserves Group of Pictures structure for video understanding.
"""

import numpy as np
import torch
from PIL import Image
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class GOPVideoData:
    """
    Data structure to hold GOP-aware video data.

    Attributes:
        i_frames: List of PIL Images for I-frames
        i_frame_indices: Original indices of I-frames in the full motion-vector timeline (at fps);
            used for timestamp calculation. These are the original positions, not indices into
            the subsampled motion_vectors array.
        motion_vectors: Motion vectors for SAMPLED GOPs only (subsampled to reduce memory).
            Shape is reduced from all video frames to only frames in sampled GOPs.
        frame_types: Frame types for sampled GOPs only (0=I, 1=P, 2=B)
        gop_boundaries: List of (start, end) tuples for each GOP, relative to the
            SUBSAMPLED motion_vectors array (not the original video timeline).
        temporal_mapping: Maps sampled frame indices to original video frame indices
        fps: FPS rate used for motion vector extraction
        total_frames: Total number of frames in the subsampled data
        video_time: Original video duration in seconds

    Note: After optimization, motion_vectors and gop_boundaries are relative to the
    subsampled data, while i_frame_indices keeps original positions for timestamps.
    """
    i_frames: List[Image.Image]
    i_frame_indices: List[int]
    motion_vectors: np.ndarray
    frame_types: np.ndarray
    gop_boundaries: List[Tuple[int, int]]
    temporal_mapping: Dict[int, int]
    fps: int
    total_frames: int
    video_time: float
    
    def get_gop_for_frame(self, frame_idx: int) -> Optional[int]:
        """Get GOP index for a given frame index (in subsampled coordinates)."""
        for gop_idx, (start, end) in enumerate(self.gop_boundaries):
            if start <= frame_idx < end:
                return gop_idx
        return None

    def get_motion_vectors_for_gop(self, gop_idx: int) -> np.ndarray:
        """Get motion vectors for a specific GOP (using subsampled data)."""
        if gop_idx >= len(self.gop_boundaries):
            return None
        start, end = self.gop_boundaries[gop_idx]
        return self.motion_vectors[start:end]

    def get_pb_frames_for_iframe(self, gop_idx: int) -> np.ndarray:
        """
        Get P/B frame motion vectors associated with an I-frame by GOP index.

        Note: After optimization, use gop_idx (0, 1, 2, ...) not original frame positions.
        """
        if gop_idx >= len(self.gop_boundaries):
            return None

        start, end = self.gop_boundaries[gop_idx]
        # Return motion vectors for P/B frames in this GOP (excluding I-frame)
        pb_mask = self.frame_types[start:end] != 0
        return self.motion_vectors[start:end][pb_mask]


class GOPVideoLoader:
    """Loader for GOP-aware video processing."""
    
    def __init__(self, motion_vector_loader, max_i_frames: int = 64, max_pb_per_i: int = 32):
        """
        Initialize GOP video loader.
        
        Args:
            motion_vector_loader: Instance of MotionVectorLoader
            max_i_frames: Maximum number of I-frames to sample
            max_pb_per_i: Maximum P/B frames per I-frame
        """
        self.mv_loader = motion_vector_loader
        self.max_i_frames = max_i_frames
        self.max_pb_per_i = max_pb_per_i
    
    def load_gop_video(
        self,
        video_path: str,
        video_reader,
        fps: int = 16,
        uniform_sample: bool = True,
        fallback_to_frames: bool = True
    ) -> Optional[GOPVideoData]:
        """
        Load video with GOP awareness.

        Args:
            video_path: Path to video file
            video_reader: Decord VideoReader or similar
            fps: FPS rate for motion vectors
            uniform_sample: Whether to uniformly sample I-frames if exceeding max
            fallback_to_frames: If True, fall back to frame-only mode when MVs fail

        Returns:
            GOPVideoData object or None if loading fails
        """
        # Get motion vectors and frame types
        mv_data = self.mv_loader.get_motion_vectors_with_types(video_path, fps)
        if mv_data is None:
            logger.warning(f"Failed to load motion vectors for {video_path}")
            if not fallback_to_frames:
                return None
            # Fallback: Create synthetic GOP data with uniform I-frame sampling
            return self._create_fallback_gop_data(video_reader, fps, uniform_sample)

        motion_vectors = mv_data['motion_vectors']
        frame_types = mv_data['frame_types']
        gop_structure = mv_data['gop_structure']

        # Debug logging for FPS mismatch issues
        logger.debug(f"Loading GOP video at {fps}fps: motion_vectors shape={motion_vectors.shape}, "
                    f"frame_types len={len(frame_types)}, "
                    f"num I-frames={len(gop_structure['i_frame_positions'])}")
        
        # Find all I-frame positions
        i_frame_positions = np.array(gop_structure['i_frame_positions'])
        num_i_frames = len(i_frame_positions)
        
        # Sample I-frames if exceeding maximum
        if num_i_frames > self.max_i_frames:
            if uniform_sample:
                # Uniform sampling
                sample_indices = np.linspace(0, num_i_frames - 1, self.max_i_frames, dtype=int)
                sampled_i_positions = i_frame_positions[sample_indices]
            else:
                # Random sampling
                sample_indices = np.random.choice(num_i_frames, self.max_i_frames, replace=False)
                sample_indices.sort()
                sampled_i_positions = i_frame_positions[sample_indices]
        else:
            sampled_i_positions = i_frame_positions
        
        # Load RGB frames for sampled I-frames
        i_frames = []
        temporal_mapping = {}
        
        for idx, i_frame_pos in enumerate(sampled_i_positions):
            # Map from motion vector frame index to original video frame index
            # This depends on the FPS rate used
            original_frame_idx = self._mv_index_to_video_index(i_frame_pos, fps, video_reader)
            
            # Load the frame
            try:
                frame = video_reader.get_batch([original_frame_idx]).asnumpy()[0]
                pil_frame = Image.fromarray(frame)
                i_frames.append(pil_frame)
                temporal_mapping[idx] = original_frame_idx
            except Exception as e:
                logger.warning(f"Failed to load I-frame at index {original_frame_idx}: {e}")
                i_frames.append(Image.new('RGB', (384, 384), color='black'))
        
        # Determine GOP boundaries for sampled I-frames
        gop_boundaries_original = []
        max_frame_idx = len(frame_types) - 1

        for i, i_pos in enumerate(sampled_i_positions):
            # Ensure i_pos is within bounds
            i_pos = min(int(i_pos), max_frame_idx)

            # Find the next I-frame position
            next_i_idx = np.where(i_frame_positions > i_pos)[0]
            if len(next_i_idx) > 0:
                next_i_pos = min(int(i_frame_positions[next_i_idx[0]]), len(frame_types))
            else:
                next_i_pos = len(frame_types)

            # Validate boundaries
            if i_pos >= next_i_pos:
                logger.warning(f"Invalid GOP boundary detected: start={i_pos} >= end={next_i_pos}")
                # Use a minimal valid boundary
                next_i_pos = min(i_pos + 1, len(frame_types))

            gop_boundaries_original.append((int(i_pos), int(next_i_pos)))

        # OPTIMIZATION: Only keep motion vectors for sampled GOPs to reduce memory
        # Instead of loading all 1893 frames' MVs, only load MVs for the sampled ~16 GOPs
        sampled_mv_list = []
        sampled_frame_types_list = []
        gop_boundaries = []  # New boundaries relative to the subsampled array
        current_offset = 0

        for start, end in gop_boundaries_original:
            gop_len = end - start
            # Extract motion vectors for this GOP only
            sampled_mv_list.append(motion_vectors[start:end])
            sampled_frame_types_list.append(frame_types[start:end])
            # New boundary is relative to the concatenated subsampled array
            gop_boundaries.append((current_offset, current_offset + gop_len))
            current_offset += gop_len

        # Concatenate only the sampled GOP motion vectors
        if sampled_mv_list:
            sampled_motion_vectors = np.concatenate(sampled_mv_list, axis=0)
            sampled_frame_types = np.concatenate(sampled_frame_types_list, axis=0)
        else:
            sampled_motion_vectors = motion_vectors[:0]  # Empty array with same shape except first dim
            sampled_frame_types = frame_types[:0]

        # Print for debugging (visible during training)
        print(f"[GOP Optimization] MVs reduced: {len(motion_vectors)} -> {len(sampled_motion_vectors)} frames "
              f"({len(sampled_i_positions)} GOPs, shape: {sampled_motion_vectors.shape})")

        # Calculate video time
        total_frames_original = len(video_reader)
        avg_fps_original = video_reader.get_avg_fps()
        video_time = total_frames_original / avg_fps_original

        return GOPVideoData(
            i_frames=i_frames,
            i_frame_indices=sampled_i_positions.tolist(),  # Keep original positions for timestamp calculation
            motion_vectors=sampled_motion_vectors,
            frame_types=sampled_frame_types,
            gop_boundaries=gop_boundaries,  # These are now 0-indexed relative to subsampled motion vectors
            temporal_mapping=temporal_mapping,
            fps=fps,
            total_frames=len(sampled_frame_types),
            video_time=video_time
        )

    def _create_fallback_gop_data(self, video_reader, fps: int, uniform_sample: bool) -> GOPVideoData:
        """
        Create fallback GOP data when motion vectors are unavailable.
        Uses uniform sampling of frames as pseudo I-frames.
        """
        total_frames = len(video_reader)
        original_fps = video_reader.get_avg_fps()
        video_time = total_frames / original_fps

        # Sample frames uniformly as pseudo I-frames
        num_samples = min(self.max_i_frames, total_frames)
        if uniform_sample:
            sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        else:
            sample_indices = np.sort(np.random.choice(total_frames, num_samples, replace=False))

        # Load sampled frames
        i_frames = []
        for idx in sample_indices:
            try:
                frame = video_reader.get_batch([idx]).asnumpy()[0]
                i_frames.append(Image.fromarray(frame))
            except Exception as e:
                logger.warning(f"Failed to load frame {idx}: {e}")
                i_frames.append(Image.new('RGB', (384, 384), color='black'))

        # Create synthetic motion vectors (zeros) and frame types (all I-frames)
        synthetic_frames = len(sample_indices)
        motion_vectors = np.zeros((synthetic_frames, 7, 7, 2), dtype=np.float32)
        frame_types = np.zeros(synthetic_frames, dtype=np.int32)  # All I-frames

        # Simple GOP boundaries (one frame per GOP)
        gop_boundaries = [(i, i+1) for i in range(synthetic_frames)]

        return GOPVideoData(
            i_frames=i_frames,
            i_frame_indices=list(range(synthetic_frames)),
            motion_vectors=motion_vectors,
            frame_types=frame_types,
            gop_boundaries=gop_boundaries,
            temporal_mapping={i: sample_indices[i] for i in range(len(sample_indices))},
            fps=fps,
            total_frames=synthetic_frames,
            video_time=video_time
        )

    def _mv_index_to_video_index(self, mv_index: int, mv_fps: int, video_reader) -> int:
        """
        Convert motion vector frame index to original video frame index.
        
        Args:
            mv_index: Frame index in motion vector array
            mv_fps: FPS rate of motion vectors
            video_reader: Video reader with original FPS info
        
        Returns:
            Corresponding frame index in original video
        """
        original_fps = video_reader.get_avg_fps()
        # Calculate time position
        time_position = mv_index / mv_fps
        # Convert to original frame index
        original_index = int(time_position * original_fps)
        # Ensure within bounds
        return min(original_index, len(video_reader) - 1)
    
    def create_hybrid_tensor(self, gop_data: GOPVideoData, image_processor) -> Dict[str, Any]:
        """
        Create hybrid tensor representation for model input.

        Args:
            gop_data: GOPVideoData object
            image_processor: Image processor for I-frames

        Returns:
            Dictionary with processed tensors
        """
        try:
            # Process I-frames
            processed_i_frames = []
            for i_frame in gop_data.i_frames:
                processed = image_processor.preprocess(i_frame, return_tensors="pt")["pixel_values"][0]
                processed_i_frames.append(processed)

            # Stack I-frames
            i_frames_tensor = torch.stack(processed_i_frames) if processed_i_frames else None
        except Exception as e:
            logger.error(f"Failed to process I-frames: {e}")
            return None
        
        # Convert motion vectors to tensor
        motion_vectors_tensor = torch.from_numpy(gop_data.motion_vectors).float()
        
        # Create frame type mask
        frame_types_tensor = torch.from_numpy(gop_data.frame_types).long()
        
        return {
            'i_frames': i_frames_tensor,  # Shape: (num_i_frames, C, H, W)
            'motion_vectors': motion_vectors_tensor,  # Shape: (total_frames, h, w, 2)
            'frame_types': frame_types_tensor,  # Shape: (total_frames,)
            'i_frame_indices': torch.tensor(gop_data.i_frame_indices),
            'gop_boundaries': gop_data.gop_boundaries,
            'video_time': gop_data.video_time,
            'fps': gop_data.fps
        }

import av
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodecFeatureExtractor:
    """Extract codec features from video files for deep learning."""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.features = defaultdict(list)
        self.metadata = {}
        
    def extract_features(self, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """Extract all available codec features from the video."""
        
        container = av.open(str(self.video_path))
        stream = container.streams.video[0]
        
        # Store video metadata
        self.metadata = {
            'codec': stream.codec_context.name,
            'width': stream.width,
            'height': stream.height,
            'fps': float(stream.average_rate),
            'bit_rate': stream.bit_rate if hasattr(stream, 'bit_rate') else None,
            'pix_fmt': stream.codec_context.pix_fmt if hasattr(stream.codec_context, 'pix_fmt') else None,
            'profile': stream.codec_context.profile if hasattr(stream.codec_context, 'profile') else None,
            'time_base': str(stream.time_base),
            'duration': float(stream.duration * stream.time_base) if stream.duration else None,
            'nb_frames': stream.frames,
            'codec_type': stream.type,
            'codec_id': stream.codec_context.codec_id if hasattr(stream.codec_context, 'codec_id') else None,
        }
        
        logger.info(f"Processing video: {self.video_path}")
        logger.info(f"Codec: {self.metadata['codec']}")
        logger.info(f"Resolution: {self.metadata['width']}x{self.metadata['height']}")
        
        frame_count = 0

        # Don't skip frames - we want to analyze all frame types (I, P, B).
        for packet in container.demux(stream):
            if max_frames and frame_count >= max_frames:
                break
                
            # Packet-level features
            self.features['packet_size'].append(packet.size)
            self.features['packet_dts'].append(packet.dts)
            self.features['packet_pts'].append(packet.pts)
            self.features['packet_duration'].append(packet.duration)
            self.features['packet_pos'].append(packet.pos)
            self.features['is_keyframe'].append(packet.is_keyframe)
            
            try:
                for frame in packet.decode():
                    frame_count += 1
                    
                    # Frame-level features
                    # Map pict_type integer to string
                    pict_type_map = {0: 'NONE', 1: 'I', 2: 'P', 3: 'B', 4: 'S', 5: 'SI', 6: 'SP', 7: 'BI'}
                    self.features['frame_type'].append(pict_type_map.get(frame.pict_type, str(frame.pict_type)))
                    self.features['frame_pts'].append(frame.pts)
                    self.features['frame_dts'].append(frame.dts)
                    self.features['frame_time'].append(frame.time)
                    self.features['key_frame'].append(frame.key_frame)
                    
                    # Try to extract additional attributes if available
                    if hasattr(frame, 'coded_picture_number'):
                        self.features['coded_picture_number'].append(frame.coded_picture_number)
                    if hasattr(frame, 'display_picture_number'):
                        self.features['display_picture_number'].append(frame.display_picture_number)
                    if hasattr(frame, 'quality'):
                        self.features['quality'].append(frame.quality)
                    if hasattr(frame, 'is_corrupt'):
                        self.features['is_corrupt'].append(frame.is_corrupt)
                    
                    # Motion vector extraction (if available)
                    # Note: Motion vectors are typically not available in most codecs
                    # They require special encoder settings to export
                    self.features['has_motion_vectors'].append(False)
                    self.features['motion_vector_count'].append(0)
                    
                    # Extract quantization parameters if available
                    if hasattr(frame, 'qp_table'):
                        qp_data = self._extract_qp_table(frame)
                        self.features['has_qp_table'].append(qp_data is not None)
                        if qp_data is not None:
                            self.features['qp_mean'].append(np.mean(qp_data))
                            self.features['qp_std'].append(np.std(qp_data))
                            self.features['qp_min'].append(np.min(qp_data))
                            self.features['qp_max'].append(np.max(qp_data))
                    else:
                        self.features['has_qp_table'].append(False)
                    
                    # Try to get DCT coefficients (codec-specific)
                    if hasattr(frame, 'dct_coeff'):
                        self.features['has_dct_coeff'].append(True)
                    else:
                        self.features['has_dct_coeff'].append(False)
                    
                    if frame_count % 100 == 0:
                        logger.info(f"Processed {frame_count} frames...")
                        
            except Exception as e:
                logger.warning(f"Error decoding packet: {e}")
                continue
        
        container.close()
        
        # Convert lists to numpy arrays and compute statistics
        self.compute_feature_statistics()
        
        return {
            'metadata': self.metadata,
            'features': dict(self.features),
            'statistics': self.statistics
        }
    
    def _extract_motion_vectors(self, motion_vectors) -> List[Dict]:
        """Extract motion vector data."""
        mv_data = []
        # This is codec-specific and might need adjustment
        # PyAV's motion vector support varies by codec
        return mv_data
    
    def _extract_qp_table(self, frame) -> Optional[np.ndarray]:
        """Extract quantization parameter table from frame."""
        # QP table extraction is codec-specific
        # This would need to be implemented based on specific codec support
        return None
    
    def compute_feature_statistics(self):
        """Compute statistics for extracted features."""
        self.statistics = {}
        
        for feature_name, values in self.features.items():
            if not values:
                continue
                
            # Convert to numpy array if possible
            try:
                arr = np.array(values)
                
                # For numeric features
                if arr.dtype in [np.int32, np.int64, np.float32, np.float64]:
                    self.statistics[feature_name] = {
                        'shape': arr.shape,
                        'dtype': str(arr.dtype),
                        'mean': float(np.mean(arr)),
                        'std': float(np.std(arr)),
                        'min': float(np.min(arr)),
                        'max': float(np.max(arr)),
                        'unique_count': len(np.unique(arr))
                    }
                # For categorical features
                else:
                    unique, counts = np.unique(arr, return_counts=True)
                    self.statistics[feature_name] = {
                        'shape': arr.shape,
                        'dtype': str(arr.dtype),
                        'unique_values': unique.tolist(),
                        'value_counts': dict(zip(unique.tolist(), counts.tolist()))
                    }
            except:
                # For non-array features
                self.statistics[feature_name] = {
                    'type': type(values[0]).__name__,
                    'count': len(values)
                }
    
    def visualize_features(self, output_dir: Path):
        """Create visualizations of extracted features."""
        output_dir.mkdir(exist_ok=True)

        if 'frame_type' in self.features:
            plt.figure(figsize=(12, 6))
            frame_types = self.features['frame_type']
            frame_type_map = {'I': 0, 'P': 1, 'B': 2}
            frame_type_numeric = [frame_type_map.get(ft[0], 3) for ft in frame_types]
            plt.scatter(range(len(frame_type_numeric)), frame_type_numeric, alpha=0.6, s=1)
            plt.yticks([0, 1, 2], ['I', 'P', 'B'])
            plt.xlabel('Frame Number')
            plt.ylabel('Frame Type')
            plt.title('Frame Types Distribution')
            plt.tight_layout()
            plt.savefig(output_dir / 'frame_types.png')
            plt.close()

        if 'packet_size' in self.features:
            plt.figure(figsize=(12, 6))
            plt.plot(self.features['packet_size'], alpha=0.7)
            plt.xlabel('Packet Number')
            plt.ylabel('Packet Size (bytes)')
            plt.title('Packet Sizes Over Time')
            plt.tight_layout()
            plt.savefig(output_dir / 'packet_sizes.png')
            plt.close()

        if 'quality' in self.features and any(q is not None for q in self.features['quality']):
            plt.figure(figsize=(12, 6))
            quality_values = [q if q is not None else 0 for q in self.features['quality']]
            plt.plot(quality_values, alpha=0.7)
            plt.xlabel('Frame Number')
            plt.ylabel('Quality')
            plt.title('Frame Quality Over Time')
            plt.tight_layout()
            plt.savefig(output_dir / 'frame_quality.png')
            plt.close()
    
    def save_results(self, output_path: Path):
        """Save extracted features and statistics to JSON."""
        results = {
            'video_path': str(self.video_path),
            'extraction_time': datetime.now().isoformat(),
            'metadata': self.metadata,
            'statistics': self.statistics,
            'feature_shapes': {k: len(v) for k, v in self.features.items()},
            'sample_features': {}
        }
        
        # Save sample of features (first 10 values)
        for feature_name, values in self.features.items():
            if values:
                sample = values[:10]
                # Convert numpy types to Python types for JSON serialization
                if isinstance(sample[0], (np.integer, np.floating)):
                    sample = [float(v) for v in sample]
                results['sample_features'][feature_name] = sample
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Test codec feature extraction on a video file."""
    import argparse
    parser = argparse.ArgumentParser(description='Extract codec features from a video file.')
    parser.add_argument('video', type=str, help='Path to the input video file')
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return
    logger.info(f"Testing with video: {video_path}")
    
    # Extract features
    extractor = CodecFeatureExtractor(video_path)
    features = extractor.extract_features(max_frames=500)  # Limit for testing
    
    # Create output directory
    output_dir = Path('codec_features_output')
    output_dir.mkdir(exist_ok=True)
    
    # Save results
    output_file = output_dir / f"{video_path.stem}_codec_features.json"
    extractor.save_results(output_file)
    
    extractor.visualize_features(output_dir)
    
    # Print summary
    print("\n=== CODEC FEATURE EXTRACTION SUMMARY ===")
    print(f"\nVideo: {video_path.name}")
    print(f"Codec: {extractor.metadata['codec']}")
    print(f"Resolution: {extractor.metadata['width']}x{extractor.metadata['height']}")
    print(f"FPS: {extractor.metadata['fps']}")
    
    print("\n=== EXTRACTED FEATURES ===")
    for feature_name, stats in extractor.statistics.items():
        print(f"\n{feature_name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
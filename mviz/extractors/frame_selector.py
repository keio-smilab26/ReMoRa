import av
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMFrameSelector:
    """
    Selects optimal frames from video for LLM processing based on codec features.
    """
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.container = None
        self.stream = None
        
    def extract_keyframes_only(self, output_dir: Path) -> List[Dict]:
        """Extract only I-frames (keyframes) for maximum efficiency."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.container = av.open(str(self.video_path))
        self.stream = self.container.streams.video[0]
        
        keyframes = []
        frame_count = 0
        
        logger.info("Extracting keyframes only...")
        
        for packet in self.container.demux(self.stream):
            if packet.is_keyframe:
                for frame in packet.decode():
                    frame_count += 1
                    
                    # Save frame
                    img = frame.to_image()
                    output_path = output_dir / f"keyframe_{frame_count:06d}.jpg"
                    img.save(output_path, quality=90)
                    
                    keyframes.append({
                        'frame_idx': frame_count,
                        'timestamp': frame.time,
                        'path': str(output_path),
                        'type': 'I'
                    })
                    
                    logger.info(f"Saved keyframe {frame_count} at {frame.time:.2f}s")
        
        self.container.close()
        return keyframes
    
    def extract_adaptive_frames(self, output_dir: Path, 
                              min_interval: float = 0.5,
                              complexity_threshold: float = 0.7) -> List[Dict]:
        """
        Extract frames adaptively based on complexity and motion.
        Uses packet size as a proxy for frame complexity.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.container = av.open(str(self.video_path))
        self.stream = self.container.streams.video[0]
        
        selected_frames = []
        last_saved_time = -min_interval
        
        # Calculate packet size statistics for normalization
        packet_sizes = []
        for packet in self.container.demux(self.stream):
            packet_sizes.append(packet.size)
        
        mean_size = np.mean(packet_sizes)
        std_size = np.std(packet_sizes)
        
        # Reset container
        self.container.seek(0)
        
        logger.info("Extracting frames based on complexity...")
        
        frame_idx = 0
        for packet in self.container.demux(self.stream):
            # Normalize packet size
            if std_size > 0:
                complexity = (packet.size - mean_size) / std_size
                complexity = 1 / (1 + np.exp(-complexity))  # Sigmoid normalization
            else:
                complexity = 0.5
            
            for frame in packet.decode():
                frame_idx += 1
                current_time = frame.time if frame.time else 0
                
                # Decision criteria
                should_save = False
                reason = ""
                
                # Always save keyframes
                if frame.key_frame:
                    should_save = True
                    reason = "keyframe"
                # Save high-complexity frames with time constraint
                elif complexity > complexity_threshold and \
                     current_time - last_saved_time >= min_interval:
                    should_save = True
                    reason = f"high_complexity_{complexity:.2f}"
                # Ensure minimum sampling rate
                elif current_time - last_saved_time >= min_interval * 2:
                    should_save = True
                    reason = "min_sampling"
                
                if should_save:
                    # Save frame
                    img = frame.to_image()
                    output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                    img.save(output_path, quality=90)
                    
                    # Map frame type
                    pict_type_map = {0: 'NONE', 1: 'I', 2: 'P', 3: 'B'}
                    frame_type = pict_type_map.get(frame.pict_type, 'U')
                    
                    selected_frames.append({
                        'frame_idx': frame_idx,
                        'timestamp': current_time,
                        'path': str(output_path),
                        'type': frame_type,
                        'complexity': float(complexity),
                        'packet_size': packet.size,
                        'reason': reason
                    })
                    
                    last_saved_time = current_time
                    logger.info(f"Saved frame {frame_idx} ({frame_type}) at {current_time:.2f}s - {reason}")
        
        self.container.close()
        return selected_frames
    
    def extract_scene_representative_frames(self, output_dir: Path,
                                          window_size: int = 10) -> List[Dict]:
        """
        Extract representative frames from each scene/GOP.
        Selects the most 'important' frame from each window.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.container = av.open(str(self.video_path))
        self.stream = self.container.streams.video[0]
        
        selected_frames = []
        window_buffer = deque(maxlen=window_size)
        
        logger.info("Extracting scene-representative frames...")
        
        frame_idx = 0
        for packet in self.container.demux(self.stream):
            for frame in packet.decode():
                frame_idx += 1
                
                # Store frame info in buffer
                pict_type_map = {0: 'NONE', 1: 'I', 2: 'P', 3: 'B'}
                frame_type = pict_type_map.get(frame.pict_type, 'U')
                
                frame_info = {
                    'frame': frame,
                    'idx': frame_idx,
                    'type': frame_type,
                    'packet_size': packet.size,
                    'is_keyframe': frame.key_frame,
                    'timestamp': frame.time if frame.time else 0
                }
                
                window_buffer.append(frame_info)
                
                # Process window when we hit a keyframe or window is full
                if frame.key_frame or len(window_buffer) == window_size:
                    if window_buffer:
                        # Select best frame from window
                        # Priority: I-frame > P-frame > largest B-frame
                        best_frame = None
                        
                        # Look for I-frame
                        for f in window_buffer:
                            if f['type'] == 'I':
                                best_frame = f
                                break
                        
                        # If no I-frame, look for P-frame with largest packet
                        if not best_frame:
                            p_frames = [f for f in window_buffer if f['type'] == 'P']
                            if p_frames:
                                best_frame = max(p_frames, key=lambda x: x['packet_size'])
                        
                        # If still none, take frame with largest packet
                        if not best_frame:
                            best_frame = max(window_buffer, key=lambda x: x['packet_size'])
                        
                        # Save the selected frame
                        img = best_frame['frame'].to_image()
                        output_path = output_dir / f"scene_{len(selected_frames):04d}.jpg"
                        img.save(output_path, quality=90)
                        
                        selected_frames.append({
                            'frame_idx': best_frame['idx'],
                            'timestamp': best_frame['timestamp'],
                            'path': str(output_path),
                            'type': best_frame['type'],
                            'packet_size': best_frame['packet_size'],
                            'window_size': len(window_buffer)
                        })
                        
                        logger.info(f"Selected {best_frame['type']}-frame from window of {len(window_buffer)} frames")
                        
                        # Clear buffer after processing
                        window_buffer.clear()
        
        self.container.close()
        return selected_frames
    
    def generate_summary_report(self, frames: List[Dict], output_path: Path):
        """Generate a summary report of the extracted frames."""
        report = {
            'video_path': str(self.video_path),
            'total_frames_extracted': len(frames),
            'extraction_method': 'codec_aware',
            'frames': frames,
            'statistics': {
                'frame_types': {},
                'avg_interval': 0,
                'compression_ratio': 0
            }
        }
        
        # Calculate statistics
        if frames:
            # Frame type distribution
            for frame in frames:
                frame_type = frame.get('type', 'U')
                report['statistics']['frame_types'][frame_type] = \
                    report['statistics']['frame_types'].get(frame_type, 0) + 1
            
            # Average time interval
            if len(frames) > 1:
                intervals = []
                for i in range(1, len(frames)):
                    intervals.append(frames[i]['timestamp'] - frames[i-1]['timestamp'])
                report['statistics']['avg_interval'] = np.mean(intervals)
            
            # Estimate compression ratio
            total_duration = frames[-1]['timestamp'] - frames[0]['timestamp']
            if total_duration > 0:
                fps = 25.0  # Assume 25 fps
                total_possible_frames = int(total_duration * fps)
                report['statistics']['compression_ratio'] = \
                    len(frames) / max(total_possible_frames, 1)
        
        # Save report
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved to {output_path}")
        return report


def main():
    """Demonstrate different frame extraction strategies."""
    import argparse
    parser = argparse.ArgumentParser(description='Demonstrate frame extraction strategies.')
    parser.add_argument('video', type=str, help='Path to the input video file')
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    selector = LLMFrameSelector(video_path)
    
    # Strategy 1: Keyframes only (maximum compression)
    logger.info("\n=== Strategy 1: Keyframes Only ===")
    keyframes_dir = Path('llm_frames_output/keyframes_only')
    keyframes = selector.extract_keyframes_only(keyframes_dir)
    selector.generate_summary_report(keyframes, keyframes_dir / 'summary.json')
    
    # Strategy 2: Adaptive extraction (balanced)
    logger.info("\n=== Strategy 2: Adaptive Extraction ===")
    adaptive_dir = Path('llm_frames_output/adaptive')
    adaptive_frames = selector.extract_adaptive_frames(adaptive_dir, 
                                                      min_interval=1.0,
                                                      complexity_threshold=0.6)
    selector.generate_summary_report(adaptive_frames, adaptive_dir / 'summary.json')
    
    # Strategy 3: Scene-representative frames
    logger.info("\n=== Strategy 3: Scene Representative ===")
    scene_dir = Path('llm_frames_output/scene_representative')
    scene_frames = selector.extract_scene_representative_frames(scene_dir, window_size=15)
    selector.generate_summary_report(scene_frames, scene_dir / 'summary.json')
    
    # Print comparison
    print("\n=== EXTRACTION COMPARISON ===")
    print(f"Original video: {video_path.name}")
    print(f"Keyframes only: {len(keyframes)} frames")
    print(f"Adaptive extraction: {len(adaptive_frames)} frames")
    print(f"Scene representative: {len(scene_frames)} frames")


if __name__ == "__main__":
    main()
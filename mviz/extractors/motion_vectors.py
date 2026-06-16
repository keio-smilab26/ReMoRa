#!/usr/bin/env python3
"""
Extract motion vectors directly from video codecs using FFmpeg.
"""

import subprocess
import json
import numpy as np
from pathlib import Path
import struct
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Optional
import logging
import tempfile
import os
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FFmpegMotionVectorExtractor:
    """Extract raw motion vectors using FFmpeg's libavcodec."""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.motion_vectors = []
        self.frame_info = []
        
    def extract_motion_vectors_raw(self, max_frames: Optional[int] = None) -> Dict:
        """
        Extract raw motion vectors using FFmpeg with special flags.
        This uses FFmpeg's ability to export motion vectors as side data.
        """
        output_file = Path('motion_vectors_raw.json')
        
        # First, let's check if FFmpeg supports motion vector export
        logger.info("Checking FFmpeg capabilities...")
        self._check_ffmpeg_capabilities()
        
        # Method 1: Extract actual motion vectors using ffmpeg with export_mvs
        logger.info("Extracting motion vectors using FFmpeg with export_mvs flag...")
        
        # First get frame information
        probe_cmd = [
            'ffprobe',
            '-show_frames',
            '-select_streams', 'v:0',
            '-show_entries', 'frame=pkt_pts_time,pkt_size,pict_type,key_frame,coded_picture_number',
            '-print_format', 'json',
            str(self.video_path)
        ]
        
        if max_frames:
            probe_cmd.extend(['-read_intervals', f'%+{max_frames}'])
        
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            frame_data = json.loads(result.stdout)
            self.frame_info = frame_data.get('frames', [])
            logger.info(f"Extracted frame information for {len(self.frame_info)} frames")
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Error extracting frame data: {e}")
        
        # Method 2: Extract motion vectors using codec analysis with 4x4 blocks
        mv_data = self._extract_with_codec_analysis()
        
        # Method 3: Try to extract motion vectors using a different approach
        mv_data_alt = self._extract_motion_vectors_alternative(max_frames)
        
        # Combine results
        final_result = {
            'frame_info': self.frame_info,
            'motion_estimation': mv_data,
            'motion_vectors_raw': mv_data_alt,
            'extraction_timestamp': str(datetime.now())
        }
        
        # Save results to JSON
        with open(output_file, 'w') as f:
            json.dump(final_result, f, indent=2)
        
        logger.info(f"Motion vector data saved to {output_file}")
        return final_result
    
    def _extract_with_codec_analysis(self) -> Dict:
        """Extract motion vectors by analyzing codec data at a lower level."""
        
        # Create a temporary file for motion vector data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Use FFmpeg to extract motion vector estimation data with 4x4 macroblock size
        cmd = [
            'ffmpeg',
            '-flags2', '+export_mvs',
            '-i', str(self.video_path),
            '-vf', 'mestimate=method=hexbs:mb_size=4:search_param=7,metadata=print:file=' + tmp_path,
            '-f', 'null',
            '-'
        ]
        
        motion_vectors = []
        try:
            logger.info("Extracting motion estimation data with 4x4 macroblock size...")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse the motion estimation output
            if Path(tmp_path).exists():
                with open(tmp_path, 'r') as f:
                    for line in f:
                        # Parse motion estimation metadata
                        if 'lavfi.mestimate' in line:
                            mv_data = self._parse_motion_vector_line(line)
                            if mv_data:
                                motion_vectors.append(mv_data)
                
                logger.info(f"Extracted {len(motion_vectors)} motion vector entries")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Motion estimation extraction failed: {e}")
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)
        
        # Return parsed motion vectors
        return {
            'motion_vectors': motion_vectors,
            'macroblock_size': 4,
            'method': 'hexbs'
        }
    
    def _extract_block_motion(self) -> Dict:
        """Extract block-level motion information."""
        
        logger.info("Extracting block-level motion data...")
        
        # Extract macroblock information
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
        
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-vf', f'signalstats=stat=tout+vrep+brng,metadata=print:file={tmp_path}',
            '-f', 'null',
            '-'
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Parse signal statistics
            block_data = []
            if Path(tmp_path).exists():
                with open(tmp_path, 'r') as f:
                    for line in f:
                        if 'lavfi.signalstats' in line:
                            block_data.append(line.strip())
                
                logger.info(f"Extracted {len(block_data)} frames of block data")
        except subprocess.CalledProcessError as e:
            logger.error(f"Block motion extraction failed: {e}")
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)
        
        return {
            'frame_count': len(self.frame_info),
            'block_data_count': len(block_data) if 'block_data' in locals() else 0
        }
    
    def extract_motion_vectors_visual(self, output_path: Optional[Path] = None) -> Path:
        """
        Extract motion vectors as a visual overlay video.
        This is the most reliable method with standard FFmpeg.
        """
        if output_path is None:
            output_path = Path('motion_vectors_visual.mp4')
        
        logger.info("Creating motion vector visualization...")
        
        # Create video with motion vectors overlaid
        cmd = [
            'ffmpeg',
            '-flags2', '+export_mvs',
            '-i', str(self.video_path),
            '-vf', 'codecview=mv=pf+bf+bb',
            '-c:v', 'libx264',
            '-crf', '18',
            '-preset', 'fast',
            str(output_path),
            '-y'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Motion vector visualization saved to {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create motion vector visualization: {e}")
            return None
    
    def extract_motion_compensated_frames(self, output_dir: Path) -> List[Path]:
        """
        Extract motion-compensated frame differences.
        This shows areas of high motion between frames.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Extracting motion-compensated frame differences...")
        
        # Extract frame differences using motion compensation
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-vf', 'tblend=all_mode=difference128',
            '-q:v', '2',
            str(output_dir / 'diff_%04d.jpg'),
            '-y'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            diff_files = sorted(output_dir.glob('diff_*.jpg'))
            logger.info(f"Extracted {len(diff_files)} motion difference frames")
            return diff_files
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract motion differences: {e}")
            return []
    
    def analyze_motion_intensity(self) -> Dict:
        """
        Analyze motion intensity using various FFmpeg filters.
        """
        logger.info("Analyzing motion intensity...")
        
        # Method 1: Scene change detection (indicates high motion)
        scene_cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-vf', 'select=gt(scene\\,0.3),showinfo',
            '-f', 'null',
            '-'
        ]
        
        try:
            result = subprocess.run(scene_cmd, capture_output=True, text=True)
            scene_changes = result.stderr.count('[Parsed_showinfo_')
            logger.info(f"Detected {scene_changes} scene changes")
        except subprocess.CalledProcessError:
            scene_changes = 0
        
        # Method 2: Motion interpolation quality (indicates motion complexity)
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
        
        mi_cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-vf', f'minterpolate=mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,metadata=print:file={tmp_path}',
            '-frames:v', '100',
            '-f', 'null',
            '-'
        ]
        
        motion_scores = []
        try:
            subprocess.run(mi_cmd, capture_output=True)
            
            # Parse motion interpolation data
            if Path(tmp_path).exists():
                with open(tmp_path, 'r') as f:
                    for line in f:
                        if 'scd_score' in line:
                            try:
                                score = float(line.split('=')[1])
                                motion_scores.append(score)
                            except:
                                pass
        except subprocess.CalledProcessError:
            pass
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)
        
        return {
            'scene_changes': scene_changes,
            'motion_scores': motion_scores,
            'avg_motion_score': np.mean(motion_scores) if motion_scores else 0
        }
    
    def extract_motion_histogram(self, output_path: Optional[Path] = None) -> np.ndarray:
        """
        Extract motion histogram data showing motion distribution.
        """
        if output_path is None:
            output_path = Path('motion_histogram.png')
        
        logger.info("Extracting motion histogram...")
        
        # Use histogram filter to analyze motion
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
        
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-vf', f'tblend=all_mode=difference,histogram,metadata=print:file={tmp_path}',
            '-frames:v', '200',
            '-f', 'null',
            '-'
        ]
        
        histogram_data = []
        try:
            subprocess.run(cmd, capture_output=True)
            
            # Parse histogram data
            if Path(tmp_path).exists():
                with open(tmp_path, 'r') as f:
                    content = f.read()
                    # Extract histogram values
                    lines = content.split('\n')
                    for line in lines:
                        if 'lavfi.histogram' in line:
                            histogram_data.append(line)
            
            logger.info(f"Extracted histogram data for {len(histogram_data)} frames")
        except subprocess.CalledProcessError as e:
            logger.error(f"Histogram extraction failed: {e}")
        finally:
            if Path(tmp_path).exists():
                os.unlink(tmp_path)
        
        return histogram_data
    
    def _parse_motion_vector_line(self, line: str) -> Optional[Dict]:
        """Parse a line of motion vector metadata from FFmpeg output."""
        try:
            # Example line format: frame:0 pts:0.000000 pts_time:0.000000
            # lavfi.mestimate.mvs_x=10,20,30 lavfi.mestimate.mvs_y=5,15,25
            
            parts = line.strip().split()
            mv_data = {}
            
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    if 'mestimate' in key:
                        # Extract motion vector components
                        if 'mvs_x' in key:
                            mv_data['mvs_x'] = [int(x) for x in value.split(',') if x]
                        elif 'mvs_y' in key:
                            mv_data['mvs_y'] = [int(y) for y in value.split(',') if y]
                        else:
                            mv_data[key.split('.')[-1]] = value
                    elif key in ['frame', 'pts', 'pts_time']:
                        try:
                            mv_data[key] = float(value) if '.' in value else int(value)
                        except:
                            mv_data[key] = value
            
            return mv_data if mv_data else None
        except Exception as e:
            logger.debug(f"Failed to parse motion vector line: {e}")
            return None
    
    def _extract_motion_vectors_alternative(self, max_frames: Optional[int] = None) -> List[Dict]:
        """Alternative method to extract motion vectors using FFmpeg's showinfo filter."""
        motion_vectors = []
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Use showinfo filter to get detailed frame information
        cmd = [
            'ffmpeg',
            '-flags2', '+export_mvs',
            '-i', str(self.video_path),
            '-vf', f'mestimate=method=hexbs:mb_size=4:search_param=7,showinfo',
            '-f', 'null',
            '-'
        ]
        
        if max_frames:
            cmd.extend(['-frames:v', str(max_frames)])
        
        try:
            logger.info("Extracting motion vectors using alternative method...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse stderr output where showinfo writes
            lines = result.stderr.split('\n')
            frame_idx = 0
            
            for line in lines:
                if '[Parsed_showinfo_' in line or '[showinfo @' in line:
                    # Extract frame information
                    frame_data = {'frame_idx': frame_idx}
                    
                    # Parse various frame attributes
                    if 'n:' in line:
                        match = re.search(r'n:\s*(\d+)', line)
                        if match:
                            frame_data['n'] = int(match.group(1))
                    
                    if 'pts:' in line:
                        match = re.search(r'pts:\s*(\d+)', line)
                        if match:
                            frame_data['pts'] = int(match.group(1))
                    
                    if 'pts_time:' in line:
                        match = re.search(r'pts_time:([\d.]+)', line)
                        if match:
                            frame_data['pts_time'] = float(match.group(1))
                    
                    motion_vectors.append(frame_data)
                    frame_idx += 1
            
            logger.info(f"Extracted {len(motion_vectors)} frames using alternative method")
        except subprocess.CalledProcessError as e:
            logger.error(f"Alternative extraction failed: {e}")
        
        return motion_vectors
    
    def _check_ffmpeg_capabilities(self):
        """Check FFmpeg build configuration for motion vector support."""
        cmd = ['ffmpeg', '-version']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            version_info = result.stdout
            
            # Check for important configurations
            if '--enable-libx264' in version_info:
                logger.info("✓ x264 support enabled")
            if '--enable-libx265' in version_info:
                logger.info("✓ x265 support enabled")
            
            # Check available filters
            filter_cmd = ['ffmpeg', '-filters']
            result = subprocess.run(filter_cmd, capture_output=True, text=True)
            filters = result.stdout
            
            important_filters = ['codecview', 'mestimate', 'minterpolate', 'signalstats']
            for f in important_filters:
                if f in filters:
                    logger.info(f"✓ {f} filter available")
                else:
                    logger.warning(f"✗ {f} filter not available")
                    
        except subprocess.CalledProcessError:
            logger.warning("Could not check FFmpeg capabilities")
    


def extract_advanced_motion_features(video_path: Path) -> Dict:
    """
    Extract advanced motion features using multiple FFmpeg techniques.
    """
    extractor = FFmpegMotionVectorExtractor(video_path)
    
    logger.info("\n=== ADVANCED MOTION FEATURE EXTRACTION ===")
    
    # 1. Extract raw motion data
    logger.info("\n1. Extracting raw motion vector data...")
    raw_data = extractor.extract_motion_vectors_raw(max_frames=200)
    
    # 2. Create visual motion vector overlay
    logger.info("\n2. Creating motion vector visualization...")
    mv_video = extractor.extract_motion_vectors_visual()
    
    # 3. Extract motion-compensated differences
    logger.info("\n3. Extracting motion-compensated frame differences...")
    diff_dir = Path('motion_diffs')
    diff_frames = extractor.extract_motion_compensated_frames(diff_dir)
    
    # 4. Analyze motion intensity
    logger.info("\n4. Analyzing motion intensity...")
    motion_analysis = extractor.analyze_motion_intensity()
    
    # 5. Extract motion histogram
    logger.info("\n6. Extracting motion histogram data...")
    histogram_data = extractor.extract_motion_histogram()
    
    # Compile results
    results = {
        'video_path': str(video_path),
        'raw_motion_data': raw_data,
        'motion_vector_video': str(mv_video) if mv_video else None,
        'motion_diff_frames': len(diff_frames),
        'motion_analysis': motion_analysis,
        'histogram_frames': len(histogram_data)
    }
    
    # Save results
    with open('ffmpeg_motion_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def create_motion_based_features():
    """
    Create motion-based feature vectors for deep learning.
    """
    logger.info("\n=== MOTION-BASED FEATURE EXTRACTION ===")
    
    # Load motion analysis results
    with open('ffmpeg_motion_analysis.json', 'r') as f:
        results = json.load(f)
    
    print("\nExtracted Motion Features:")
    print(f"1. Scene changes detected: {results['motion_analysis']['scene_changes']}")
    print(f"2. Average motion score: {results['motion_analysis']['avg_motion_score']:.3f}")
    print(f"3. Motion difference frames: {results['motion_diff_frames']}")
    print(f"4. Histogram data frames: {results['histogram_frames']}")
    
    print("\nFeature Vector Components:")
    print("- Temporal motion variance (from differences)")
    print("- Scene change frequency")
    print("- Motion intensity distribution")
    print("- Block-level motion statistics")
    
    print("\nGenerated Outputs:")
    print(f"- Motion vector visualization: {results['motion_vector_video']}")
    print(f"- Motion differences: motion_diffs/")


def main():
    """Main demonstration of FFmpeg motion vector extraction."""
    import argparse
    parser = argparse.ArgumentParser(description='Extract motion vectors from a video file.')
    parser.add_argument('video', type=str, help='Path to the input video file')
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return

    # Extract all motion features
    results = extract_advanced_motion_features(video_path)
    
    # Create feature vectors
    create_motion_based_features()
    
    print("\n=== MOTION VECTOR EXTRACTION COMPLETE ===")
    print("\nKey Findings:")
    print("1. Motion vectors can be visualized using codecview filter")
    print("2. Motion differences show actual pixel changes")
    print("3. Scene detection identifies high-motion segments")
    print("4. Multiple motion metrics can guide frame selection")
    print("\nUse these features to:")
    print("- Select frames with high motion for detailed analysis")
    print("- Skip redundant frames in low-motion segments")
    print("- Create motion-aware embeddings for LLMs")


if __name__ == "__main__":
    main()
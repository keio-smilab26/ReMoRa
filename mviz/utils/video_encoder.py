#!/usr/bin/env python3
"""
Video re-encoding utilities for H.264 codec with specific GOP structure.

This module provides functionality to re-encode videos with:
- H.264 codec (software or NVIDIA GPU accelerated)
- Variable GOP structure with configurable maximum keyframe interval
- Minimum GOP size of 1 frame (allows scene-based I-frame insertion)
- Scene change detection enabled for adaptive GOP structure
- Configurable frame rate
- Consistent format for motion vector extraction
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_nvenc_available() -> bool:
    """Check if NVIDIA NVENC hardware encoder is available."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=10
        )
        return 'h264_nvenc' in result.stdout
    except Exception:
        return False


# Cache the NVENC availability check
_NVENC_AVAILABLE = None

def is_nvenc_available() -> bool:
    """Check if NVENC is available (cached)."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None:
        _NVENC_AVAILABLE = check_nvenc_available()
        if _NVENC_AVAILABLE:
            logger.info("NVIDIA NVENC hardware encoder detected")
        else:
            logger.info("NVENC not available, using software encoder")
    return _NVENC_AVAILABLE


class VideoEncoder:
    """Re-encode videos with specific H.264 settings for motion vector extraction."""

    def __init__(self, temp_dir: Optional[Path] = None, use_gpu: bool = True):
        """
        Initialize video encoder.

        Args:
            temp_dir: Directory for temporary files (default: system temp)
            use_gpu: Whether to use GPU acceleration if available (default: True)
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.temp_dir.mkdir(exist_ok=True)
        self.use_gpu = use_gpu and is_nvenc_available()

    def reencode_video(self, input_path: Path, output_path: Path,
                      target_fps: float = 4.0, keyframe_interval: int = 8,
                      resolution: Optional[Tuple[int, int]] = None,
                      crf: int = 23, bframes: int = 2) -> bool:
        """
        Re-encode video with H.264 codec and variable GOP structure.
        Uses NVIDIA NVENC if available for hardware acceleration.

        Args:
            input_path: Path to input video file
            output_path: Path to output re-encoded video
            target_fps: Target frame rate (default: 4.0 fps)
            keyframe_interval: Maximum keyframe interval/GOP size (default: 8)
            resolution: Optional target resolution (width, height)
            crf: Constant Rate Factor for quality (default: 23, lower=better quality)
            bframes: Maximum number of B-frames (default: 2, use 0 for P-only)

        Returns:
            True if re-encoding successful, False otherwise
        """
        if self.use_gpu:
            return self._reencode_nvenc(input_path, output_path, target_fps,
                                        keyframe_interval, resolution, crf, bframes)
        else:
            return self._reencode_software(input_path, output_path, target_fps,
                                           keyframe_interval, resolution, crf, bframes)

    def _reencode_nvenc(self, input_path: Path, output_path: Path,
                        target_fps: float, keyframe_interval: int,
                        resolution: Optional[Tuple[int, int]],
                        crf: int, bframes: int) -> bool:
        """Re-encode using NVIDIA NVENC hardware encoder (5-10x faster)."""
        try:
            logger.info(f"Re-encoding {input_path.name} with NVENC (GPU), max GOP={keyframe_interval}, fps={target_fps}")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Build video filter string
            video_filters = [f'fps={target_fps}']
            if resolution:
                width, height = resolution
                video_filters.append(f'scale={width}:{height}')

            # NVENC quality - use constant quantizer
            cq_value = crf

            # Simpler NVENC command (software decode, hardware encode)
            # This is more compatible across different ffmpeg builds
            cmd = [
                'ffmpeg',
                '-y',
                '-i', str(input_path),

                # Video filters first (CPU-based)
                '-vf', ','.join(video_filters),

                # NVIDIA NVENC encoder
                '-c:v', 'h264_nvenc',
                '-preset', 'p1',  # p1 = fastest NVENC preset
                '-rc', 'vbr',     # Variable bitrate mode
                '-cq', str(cq_value),  # Quality level

                # GOP structure
                '-g', str(keyframe_interval),
                '-bf', str(bframes),

                # Output format
                '-pix_fmt', 'yuv420p',
                '-an',

                str(output_path)
            ]

            logger.debug(f"FFmpeg NVENC command: {' '.join(cmd)}")

            logger.info(f"NVENC command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                logger.info(f"Successfully re-encoded with NVENC: {output_path.name}")
                return True
            else:
                logger.error(f"NVENC failed with error:\n{result.stderr}")
                # Don't fall back to software - just fail so we can debug
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"NVENC re-encoding timeout for {input_path.name}")
            return False
        except Exception as e:
            logger.warning(f"NVENC error, falling back to software: {e}")
            return self._reencode_software(input_path, output_path, target_fps,
                                           keyframe_interval, resolution, crf, bframes)

    def _reencode_software(self, input_path: Path, output_path: Path,
                           target_fps: float, keyframe_interval: int,
                           resolution: Optional[Tuple[int, int]],
                           crf: int, bframes: int) -> bool:
        """Re-encode using software libx264 encoder with ultrafast preset."""
        try:
            logger.info(f"Re-encoding {input_path.name} with libx264 (CPU), max GOP={keyframe_interval}, fps={target_fps}")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            video_filters = [f'fps={target_fps}']
            if resolution:
                width, height = resolution
                video_filters.append(f'scale={width}:{height}')

            # Build FFmpeg command - use ultrafast preset for speed
            cmd = [
                'ffmpeg',
                '-y',
                '-i', str(input_path),

                '-c:v', 'libx264',
                '-preset', 'ultrafast',  # Fastest software encoding
                '-crf', str(crf),

                # GOP structure
                '-g', str(keyframe_interval),
                '-keyint_min', '1',
                '-sc_threshold', '40',

                '-vf', ','.join(video_filters),

                '-x264-params', f'bframes={bframes}:b-adapt=2:direct=spatial:scenecut=40',
                '-bf', str(bframes),

                '-an',

                str(output_path)
            ]

            logger.debug(f"FFmpeg software command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                logger.info(f"Successfully re-encoded: {output_path.name}")
                return True
            else:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Re-encoding timeout for {input_path.name}")
            return False
        except Exception as e:
            logger.error(f"Re-encoding error: {e}")
            return False

    def reencode_fixed_frames(self, input_path: Path, output_path: Path,
                               n_frames: int = 64, keyframe_interval: int = 8,
                               resolution: Optional[Tuple[int, int]] = None,
                               crf: int = 23, bframes: int = 2) -> bool:
        """
        Re-encode video to a fixed number of output frames.

        Instead of fixed FPS, this calculates the FPS needed to get exactly n_frames.

        Args:
            input_path: Path to input video file
            output_path: Path to output re-encoded video
            n_frames: Target number of output frames (default: 64)
            keyframe_interval: Maximum keyframe interval/GOP size (default: 8)
            resolution: Optional target resolution (width, height)
            crf: Constant Rate Factor for quality (default: 23)
            bframes: Maximum number of B-frames (default: 2)

        Returns:
            True if re-encoding successful, False otherwise
        """
        try:
            # Get video duration using ffprobe
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                   '-show_format', str(input_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"ffprobe failed for {input_path}")
                return False

            import json
            data = json.loads(result.stdout)
            duration = float(data.get('format', {}).get('duration', 0))

            if duration <= 0:
                logger.error(f"Could not get duration for {input_path}")
                return False

            # Calculate FPS to get exactly n_frames
            target_fps = n_frames / duration

            logger.info(f"Re-encoding {input_path.name} to {n_frames} frames (fps={target_fps:.2f})")

            # Use the existing reencode_video method
            return self.reencode_video(
                input_path=input_path,
                output_path=output_path,
                target_fps=target_fps,
                keyframe_interval=keyframe_interval,
                resolution=resolution,
                crf=crf,
                bframes=bframes
            )

        except Exception as e:
            logger.error(f"Error in reencode_fixed_frames: {e}")
            return False

    def reencode_scene_adaptive(self, input_path: Path, output_path: Path,
                                 target_fps: float = 16.0,
                                 max_gop: int = 32,
                                 max_iframes: int = 64,
                                 resolution: Tuple[int, int] = (384, 384),
                                 crf: int = 23, bframes: int = 2,
                                 fast_mode: bool = False) -> bool:
        """Re-encode at fixed fps (default 16) with scene-adaptive GOP up to
        ``max_gop`` frames, capped at ``max_iframes`` I-frames; quality mode
        keeps 4×4 partitions for fine-grained motion vectors.

        ``fast_mode=True`` swaps in a faster x264 preset that forces 16×16
        partitions only.
        """
        try:
            # Get video duration
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                   '-show_format', str(input_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.error(f"ffprobe failed for {input_path}")
                return False

            import json
            data = json.loads(result.stdout)
            duration = float(data.get('format', {}).get('duration', 0))

            if duration <= 0:
                logger.error(f"Could not get duration for {input_path}")
                return False

            # Calculate frames at target fps
            frames_at_target_fps = duration * target_fps

            # Max frames to stay within max_iframes budget
            # With scene-adaptive GOP, we get ~1 I-frame per max_gop frames on average
            # But scene changes can create more, so we use a conservative estimate
            max_frames = max_iframes * max_gop

            # Adjust fps if video would exceed max_iframes
            if frames_at_target_fps > max_frames:
                adjusted_fps = max_frames / duration
                logger.info(f"Adjusting fps from {target_fps} to {adjusted_fps:.2f} to cap at {max_iframes} I-frames")
                actual_fps = adjusted_fps
            else:
                actual_fps = target_fps

            output_path.parent.mkdir(parents=True, exist_ok=True)

            width, height = resolution

            if fast_mode:
                # SPEED-OPTIMIZED encoding (maximum speed)
                # Prioritizes encoding speed over compression efficiency
                # Target: faster than Uniform 128, slower than Uniform 64
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-threads', '0',  # Use all CPU threads
                    '-i', str(input_path),

                    # Video filters
                    '-vf', f'fps={actual_fps},scale={width}:{height}:flags=fast_bilinear',

                    # H.264 codec with libx264
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-tune', 'zerolatency',  # Fastest encoding (no lookahead)
                    '-crf', str(crf),

                    # Scene-adaptive GOP structure
                    '-g', str(max_gop),
                    '-keyint_min', '1',
                    '-sc_threshold', '40',

                    # Maximum speed x264 params:
                    # - bframes=0: No B-frames
                    # - subme=0: Fastest subpixel ME
                    # - me=dia: Diamond search (fastest)
                    # - ref=1: Single reference frame
                    # - no-cabac: Use CAVLC
                    # - no-deblock: Skip deblocking
                    # - partitions=none: 16x16 only
                    # - aq-mode=0: Disable adaptive quantization
                    # - no-mbtree: Disable MB-tree lookahead
                    # - trellis=0: Disable trellis optimization
                    # - weightp=0: Disable weighted prediction
                    # - rc-lookahead=0: No lookahead
                    '-x264-params', 'bframes=0:subme=0:me=dia:ref=1:no-cabac:no-deblock:partitions=none:aq-mode=0:no-mbtree:trellis=0:weightp=0:rc-lookahead=0:scenecut=40',

                    '-bf', '0',
                    '-threads', '0',  # Encoding threads
                    '-pix_fmt', 'yuv420p',
                    '-an',

                    str(output_path)
                ]
            else:
                # QUALITY-OPTIMIZED encoding (original)
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', str(input_path),

                    # Video filters
                    '-vf', f'fps={actual_fps},scale={width}:{height}',

                    # H.264 codec with libx264
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', str(crf),

                    # Scene-adaptive GOP structure
                    '-g', str(max_gop),
                    '-keyint_min', '1',
                    '-sc_threshold', '40',

                    # Quality params with fine-grained MVs
                    '-x264-params', f'bframes={bframes}:b-adapt=2:scenecut=40:partitions=all:direct=auto:ref=1',

                    '-bf', str(bframes),
                    '-pix_fmt', 'yuv420p',
                    '-an',

                    str(output_path)
                ]

            logger.info(f"Scene-adaptive encoding ({'fast' if fast_mode else 'quality'}): {input_path.name} -> {actual_fps:.1f}fps, max_gop={max_gop}")
            logger.debug(f"Command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                logger.info(f"Successfully encoded: {output_path.name}")
                return True
            else:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Encoding timeout for {input_path.name}")
            return False
        except Exception as e:
            logger.error(f"Encoding error: {e}")
            return False

    def batch_reencode(self, input_dir: Path, output_dir: Path,
                      target_fps: float = 4.0, keyframe_interval: int = 8,
                      resolution: Optional[Tuple[int, int]] = None,
                      crf: int = 23, bframes: int = 2,
                      extensions: list = None) -> dict:
        """
        Batch re-encode all videos in a directory with variable GOP structure.

        Args:
            input_dir: Directory containing input videos
            output_dir: Directory for re-encoded videos
            target_fps: Target frame rate (default: 4.0 fps)
            keyframe_interval: Maximum keyframe interval/GOP size (default: 8)
                             Actual GOP sizes will vary from 1 to this value based on scene changes
            resolution: Optional target resolution (width, height)
            crf: Constant Rate Factor for quality (default: 23, lower=better quality)
            bframes: Maximum number of B-frames (default: 2, use 0 for P-only)
            extensions: List of video file extensions to process

        Returns:
            Dictionary with processing results
        """
        if extensions is None:
            extensions = ['mp4', 'avi', 'mov', 'mkv', 'webm']
            
        # Find all video files
        video_files = []
        for ext in extensions:
            video_files.extend(input_dir.glob(f'*.{ext}'))
            video_files.extend(input_dir.glob(f'**/*.{ext}'))  # Recursive
        
        video_files = sorted(set(video_files))  # Remove duplicates
        
        if not video_files:
            logger.warning(f"No video files found in {input_dir}")
            return {'successful': [], 'failed': []}
        
        logger.info(f"Found {len(video_files)} videos to re-encode")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        successful = []
        failed = []
        
        for video_path in video_files:
            # Generate output path (maintain directory structure)
            relative_path = video_path.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix('.mp4')  # Standardize to .mp4
            
            # Re-encode video
            success = self.reencode_video(
                video_path,
                output_path,
                target_fps=target_fps,
                keyframe_interval=keyframe_interval,
                resolution=resolution,
                crf=crf,
                bframes=bframes
            )
            
            if success:
                successful.append({
                    'input': str(video_path),
                    'output': str(output_path),
                    'fps': target_fps,
                    'gop_size': keyframe_interval
                })
            else:
                failed.append({
                    'input': str(video_path),
                    'error': 'Re-encoding failed'
                })
        
        logger.info(f"Re-encoding complete: {len(successful)} successful, {len(failed)} failed")
        
        return {
            'successful': successful,
            'failed': failed,
            'settings': {
                'fps': target_fps,
                'keyframe_interval': keyframe_interval,
                'resolution': resolution,
                'crf': crf,
                'bframes': bframes
            }
        }
    
    def verify_encoding(self, video_path: Path) -> dict:
        """
        Verify the encoding settings of a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video properties
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                '-show_format',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                # Extract video stream info
                video_stream = None
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        video_stream = stream
                        break
                
                if video_stream:
                    return {
                        'codec': video_stream.get('codec_name'),
                        'width': video_stream.get('width'),
                        'height': video_stream.get('height'),
                        'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                        'duration': float(video_stream.get('duration', 0)),
                        'bitrate': video_stream.get('bit_rate'),
                        'profile': video_stream.get('profile')
                    }
            
            return {'error': 'Could not extract video information'}
            
        except Exception as e:
            return {'error': str(e)}


def main():
    """Test video encoding functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Re-encode videos with H.264 GOP structure")
    parser.add_argument('input', help='Input video file or directory')
    parser.add_argument('output', help='Output video file or directory')
    parser.add_argument('--fps', type=float, default=4.0, help='Target frame rate (default: 4.0)')
    parser.add_argument('--gop', type=int, default=8, help='Maximum GOP size/keyframe interval (min=1, default max=8)')
    parser.add_argument('--resolution', nargs=2, type=int, help='Target resolution (width height)')
    parser.add_argument('--crf', type=int, default=23, help='Constant Rate Factor (default: 23, lower=better quality)')
    parser.add_argument('--bframes', type=int, default=2, help='Maximum B-frames (default: 2, use 0 for P-only)')
    parser.add_argument('--verify', action='store_true', help='Verify encoding after completion')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    resolution = tuple(args.resolution) if args.resolution else None
    
    encoder = VideoEncoder()
    
    if input_path.is_file():
        # Single file encoding
        success = encoder.reencode_video(
            input_path,
            output_path,
            target_fps=args.fps,
            keyframe_interval=args.gop,
            resolution=resolution,
            crf=args.crf,
            bframes=args.bframes
        )
        
        if success and args.verify:
            props = encoder.verify_encoding(output_path)
            print(f"\nEncoded video properties:")
            for key, value in props.items():
                print(f"  {key}: {value}")
                
    else:
        # Batch encoding
        results = encoder.batch_reencode(
            input_path,
            output_path,
            target_fps=args.fps,
            keyframe_interval=args.gop,
            resolution=resolution,
            crf=args.crf,
            bframes=args.bframes
        )
        
        print(f"\nBatch encoding results:")
        print(f"  Successful: {len(results['successful'])}")
        print(f"  Failed: {len(results['failed'])}")
        
        if args.verify and results['successful']:
            print(f"\nVerifying first encoded video...")
            first_output = Path(results['successful'][0]['output'])
            props = encoder.verify_encoding(first_output)
            for key, value in props.items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
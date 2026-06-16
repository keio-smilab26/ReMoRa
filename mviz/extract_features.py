#!/usr/bin/env python3
"""
Main entry point for video feature extraction pipeline.

Supports multiple extraction modes:
- codec: Extract frame types, packet sizes, and codec metadata
- motion: Extract motion vectors (both estimated and codec-level)
- frames: Extract optimal frames for LLM processing
- all: Extract all features
"""

from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_codec_features(video_path: str, output_dir: Path):
    """Extract codec-level features."""
    logger.info("Extracting codec features...")
    from mviz.extractors.codec_features import CodecFeatureExtractor
    
    extractor = CodecFeatureExtractor(video_path)
    features = extractor.extract_features()
    
    # Save results
    output_file = output_dir / 'features' / 'codec_features.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    extractor.save_results(output_file)
    
    logger.info(f"Codec features saved to {output_file}")
    return features


def extract_motion_vectors(video_path: str, output_dir: Path, use_codec: bool = True):
    """Extract motion vectors (codec or estimated)."""
    logger.info("Extracting motion vectors...")
    
    if use_codec:
        # Use actual codec motion vectors
        try:
            from mviz.extractors.codec_mvs import extract_motion_vectors
            
            mv_output = output_dir / 'features' / 'motion_vectors'
            results = extract_motion_vectors(video_path, str(mv_output))
            
            logger.info(f"Codec motion vectors saved to {mv_output}")
            return results
            
        except FileNotFoundError as e:
            logger.warning(f"Codec MV extraction failed: {e}")
            logger.info("Falling back to estimated motion vectors...")
            use_codec = False
    
    if not use_codec:
        # Use FFmpeg visualization-based extraction
        from mviz.extractors.motion_vectors import FFmpegMotionVectorExtractor
        
        extractor = FFmpegMotionVectorExtractor(video_path)
        visual_path = extractor.extract_motion_vectors_visual()
        
        logger.info(f"Motion vector visualization saved to {visual_path}")
        return {'visualization': str(visual_path)}


def extract_optimal_frames(video_path: str, output_dir: Path):
    """Extract optimal frames for LLM processing."""
    logger.info("Extracting optimal frames...")
    from mviz.extractors.frame_selector import LLMFrameSelector
    
    selector = LLMFrameSelector(video_path)
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    extracted = selector.extract_adaptive_frames(frames_dir)
    logger.info(f"Extracted {len(extracted)} frames to {frames_dir}")
    
    return extracted


def main():
    parser = argparse.ArgumentParser(
        description='Video Feature Extraction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all features
  python extract_features.py video.mp4
  
  # Extract only codec features
  python extract_features.py video.mp4 --mode codec
  
  # Extract codec motion vectors with specific block sizes
  python extract_features.py video.mp4 --mode motion --block-sizes 4,8,16
  
  # Extract optimal frames for LLM
  python extract_features.py video.mp4 --mode frames
        """
    )
    
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('--mode', choices=['codec', 'motion', 'frames', 'all'], 
                        default='all', help='Extraction mode')
    parser.add_argument('--output', type=str, default='outputs/', 
                        help='Output directory')
    parser.add_argument('--block-sizes', type=str, default='4,8,16',
                        help='Block sizes for motion vectors (comma-separated)')
    parser.add_argument('--use-codec-mv', action='store_true', default=True,
                        help='Use codec motion vectors (requires extract_mvs_ffmpeg)')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    
    args = parser.parse_args()
    
    # Parse block sizes
    block_sizes = [int(x) for x in args.block_sizes.split(',')]
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Store results
    results = {}
    
    # Extract features based on mode
    if args.mode in ['codec', 'all']:
        results['codec'] = extract_codec_features(args.video, output_dir)
    
    if args.mode in ['motion', 'all']:
        results['motion'] = extract_motion_vectors(
            args.video, output_dir, use_codec=args.use_codec_mv
        )
        
        # Create visualization if requested
        if args.visualize and args.use_codec_mv:
            try:
                from mviz.visualizers.residual_grid import visualize_motion_residuals

                mv_path = output_dir / 'features' / 'motion_vectors' / f'motion_vectors_{block_sizes[0]}x{block_sizes[0]}.npy'
                if mv_path.exists():
                    viz_path = output_dir / 'visualizations' / f'motion_residuals_{block_sizes[0]}x{block_sizes[0]}.png'
                    visualize_motion_residuals(args.video, str(mv_path), str(viz_path))
                    logger.info(f"Visualization saved to {viz_path}")

            except Exception as e:
                logger.warning(f"Visualization failed: {e}")

            try:
                from mviz.visualizers.gop_visualization import create_motion_overlay_video

                video_out = output_dir / 'visualizations' / 'motion_overlay.mp4'
                create_motion_overlay_video(args.video, str(video_out))

            except Exception as e:
                logger.warning(f"Overlay video failed: {e}")
    
    if args.mode in ['frames', 'all']:
        results['frames'] = extract_optimal_frames(args.video, output_dir)
    
    # Summary
    logger.info("\n=== Feature Extraction Complete ===")
    logger.info(f"Output directory: {output_dir}")
    
    if 'codec' in results:
        logger.info("✓ Codec features extracted")
    if 'motion' in results:
        logger.info("✓ Motion vectors extracted")
    if 'frames' in results:
        logger.info("✓ Optimal frames extracted")


if __name__ == "__main__":
    main()

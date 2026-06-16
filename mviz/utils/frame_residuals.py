import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import cv2
import shutil
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameResidualAnalyzer:
    """Analyze and visualize frame residuals and motion vectors in video files."""
    
    def __init__(self, video_path: str, output_dir: str = "output", frames_json: str = None):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load frame metadata if provided
        if frames_json is not None:
            with open(frames_json, "r") as f:
                self.frame_data = json.load(f)
            self.frames = [f for f in self.frame_data["frames"] if f["media_type"] == "video"]
        else:
            self.frame_data = None
            self.frames = []

        # Get video dimensions
        self.width, self.height = self._get_video_dimensions()
    
    def _get_video_dimensions(self) -> Tuple[int, int]:
        """Extract video dimensions using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", str(self.video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        
        for stream in info["streams"]:
            if stream["codec_type"] == "video":
                return int(stream["width"]), int(stream["height"])
        
        raise ValueError("Could not determine video dimensions")
    
    def extract_motion_vectors(self) -> None:
        """Extract frames with motion vector overlay."""
        logger.info("Extracting motion vectors...")
        mv_dir = self.output_dir / "motion_vectors"
        mv_dir.mkdir(exist_ok=True)
        
        cmd = [
            "ffmpeg", "-flags2", "+export_mvs", "-i", str(self.video_path),
            "-vf", "codecview=mv=pf+bf+bb:qp=true:mv_type=fp+bp:block=true",
            str(mv_dir / "mv_%04d.png"),
            "-y"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    def extract_frame_differences(self) -> None:
        """Extract and compute frame differences for P/B frames."""
        logger.info("Computing frame differences...")
        diff_dir = self.output_dir / "differences"
        diff_dir.mkdir(exist_ok=True)
        
        # Extract all frames
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        cmd = [
            "ffmpeg", "-i", str(self.video_path),
            str(frames_dir / "frame_%04d.png"),
            "-y"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Compute differences
        prev_frame = None
        last_iframe = None
        
        for i, frame_info in enumerate(self.frames):
            frame_path = frames_dir / f"frame_{i+1:04d}.png"
            if not frame_path.exists():
                continue
            
            current_frame = cv2.imread(str(frame_path))
            if current_frame is None:
                continue
                
            frame_type = frame_info["pict_type"]
            
            if frame_type == "I":
                last_iframe = current_frame.copy()
                prev_frame = current_frame.copy()
            elif frame_type in ["P", "B"] and last_iframe is not None:
                # P/B frames encode differences from reference frames
                # For visualization, we'll show difference from last I-frame
                diff = cv2.absdiff(current_frame, last_iframe)
                
                # Save difference visualization
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                
                # Heatmap
                heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
                cv2.imwrite(str(diff_dir / f"heat_{i+1:04d}_{frame_type}.png"), heatmap)
                
                # Threshold
                _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
                cv2.imwrite(str(diff_dir / f"thresh_{i+1:04d}_{frame_type}.png"), thresh)
                
                # Also save difference from previous frame (actual residual is closer to this)
                if prev_frame is not None:
                    prev_diff = cv2.absdiff(current_frame, prev_frame)
                    prev_diff_gray = cv2.cvtColor(prev_diff, cv2.COLOR_BGR2GRAY)
                    prev_heatmap = cv2.applyColorMap(prev_diff_gray, cv2.COLORMAP_JET)
                    cv2.imwrite(str(diff_dir / f"prev_heat_{i+1:04d}_{frame_type}.png"), prev_heatmap)
                
                prev_frame = current_frame.copy()
            else:
                prev_frame = current_frame.copy()
    
    def extract_yuv_planes(self) -> Dict[str, Dict]:
        """Extract and analyze YUV planes for different frame types."""
        logger.info("Extracting YUV planes...")
        yuv_dir = self.output_dir / "yuv_analysis"
        yuv_dir.mkdir(exist_ok=True)
        
        # Extract raw YUV
        yuv_file = yuv_dir / "raw_video.yuv"
        cmd = [
            "ffmpeg", "-i", str(self.video_path),
            "-c:v", "rawvideo", "-pix_fmt", "yuv420p",
            str(yuv_file), "-y"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Analyze YUV for different frame types
        frame_size = self.width * self.height * 3 // 2
        examples = {"I": None, "P": None, "B": None}
        
        with open(yuv_file, "rb") as f:
            for i, frame_info in enumerate(self.frames[:30]):
                ftype = frame_info["pict_type"]
                if examples[ftype] is None:
                    yuv_data = f.read(frame_size)
                    if not yuv_data:
                        break
                    
                    # Extract planes
                    y_size = self.width * self.height
                    u_size = v_size = y_size // 4
                    
                    y_plane = np.frombuffer(yuv_data[:y_size], dtype=np.uint8).reshape(self.height, self.width)
                    u_plane = np.frombuffer(yuv_data[y_size:y_size+u_size], dtype=np.uint8).reshape(self.height//2, self.width//2)
                    v_plane = np.frombuffer(yuv_data[y_size+u_size:], dtype=np.uint8).reshape(self.height//2, self.width//2)
                    
                    examples[ftype] = {
                        "index": i,
                        "y": y_plane,
                        "u": u_plane,
                        "v": v_plane,
                        "info": frame_info
                    }
                else:
                    f.seek(frame_size, 1)
        
        # Clean up large YUV file
        yuv_file.unlink()
        
        return examples
    
    def create_comprehensive_visualization(self) -> None:
        """Create a comprehensive visualization of frame residuals."""
        logger.info("Creating comprehensive visualization...")
        
        # Extract all data
        self.extract_motion_vectors()
        self.extract_frame_differences()
        yuv_examples = self.extract_yuv_planes()
        
        # Create main visualization
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle("Video Frame Residual Analysis", fontsize=20, y=0.98)
        
        # Find examples of each frame type
        examples = self._find_frame_examples()
        
        # Create grid
        gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.2)
        
        for col, (ftype, idx) in enumerate(examples.items()):
            if idx is None:
                continue
            
            # Original frame
            ax1 = fig.add_subplot(gs[0, col])
            frame_path = self.output_dir / "frames" / f"frame_{idx+1:04d}.png"
            if frame_path.exists():
                img = Image.open(frame_path)
                ax1.imshow(img)
                ax1.set_title(f"{ftype}-Frame #{idx}\nSize: {self.frames[idx]['pkt_size']} bytes")
                ax1.axis('off')
            
            # Motion vectors
            ax2 = fig.add_subplot(gs[1, col])
            mv_path = self.output_dir / "motion_vectors" / f"mv_{idx+1:04d}.png"
            if mv_path.exists():
                mv_img = Image.open(mv_path)
                ax2.imshow(mv_img)
                ax2.set_title("Motion Vectors & Blocks")
                ax2.axis('off')
            
            if ftype in ["P", "B"]:
                # Difference heatmap
                ax3 = fig.add_subplot(gs[2, col])
                heat_path = self.output_dir / "differences" / f"heat_{idx+1:04d}_{ftype}.png"
                if heat_path.exists():
                    heat_img = Image.open(heat_path)
                    ax3.imshow(heat_img)
                    ax3.set_title("Difference Heatmap")
                    ax3.axis('off')
                else:
                    ax3.text(0.5, 0.5, "No heatmap available", ha='center', va='center')
                    ax3.axis('off')
                
                # Threshold
                ax4 = fig.add_subplot(gs[3, col])
                thresh_path = self.output_dir / "differences" / f"thresh_{idx+1:04d}_{ftype}.png"
                if thresh_path.exists():
                    thresh_img = Image.open(thresh_path)
                    ax4.imshow(thresh_img, cmap='gray')
                    ax4.set_title("Motion Areas")
                    ax4.axis('off')
                else:
                    ax4.text(0.5, 0.5, "No threshold data", ha='center', va='center')
                    ax4.axis('off')
            else:
                # For I-frames, show explanation instead of empty plots
                ax_combined = fig.add_subplot(gs[2:4, col])
                ax_combined.text(0.5, 0.5, 
                    "I-frames are keyframes\n\n" +
                    "• Fully encoded (no residuals)\n" +
                    "• No motion compensation\n" +
                    "• Largest file size\n" +
                    "• Used as reference for P/B frames",
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
                ax_combined.axis('off')
            
            # YUV info
            if ftype in yuv_examples and yuv_examples[ftype]:
                ax5 = fig.add_subplot(gs[4, col])
                yuv_data = yuv_examples[ftype]
                y_mean = np.mean(yuv_data["y"])
                y_std = np.std(yuv_data["y"])
                
                info_text = f"Frame Type: {ftype}\n"
                info_text += f"Y-plane stats:\n"
                info_text += f"Mean: {y_mean:.1f}\n"
                info_text += f"Std: {y_std:.1f}\n\n"
                
                if ftype == "I":
                    info_text += "Intra-coded\n(No motion compensation)"
                elif ftype == "P":
                    info_text += "Forward prediction\nfrom previous frame"
                else:
                    info_text += "Bidirectional prediction\nfrom past & future frames"
                
                ax5.text(0.1, 0.5, info_text, ha='left', va='center',
                        fontsize=11, family='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
                ax5.axis('off')
        
        plt.savefig(self.output_dir / "frame_residual_analysis.png", dpi=200, bbox_inches='tight')
        plt.close()
        
        # Create statistics plot
        self._create_statistics_plot()
        
        logger.info(f"Analysis complete. Results saved to {self.output_dir}/")
    
    def _find_frame_examples(self) -> Dict[str, Optional[int]]:
        """Find good examples of each frame type."""
        examples = {"I": None, "P": None, "B": None}
        
        # First find an I-frame
        first_iframe_idx = None
        for i, frame in enumerate(self.frames):
            if frame["pict_type"] == "I":
                first_iframe_idx = i
                examples["I"] = i
                break
        
        # Then find P and B frames after the I-frame
        if first_iframe_idx is not None:
            for i, frame in enumerate(self.frames[first_iframe_idx:], start=first_iframe_idx):
                ftype = frame["pict_type"]
                if examples[ftype] is None and i > first_iframe_idx + 5:
                    examples[ftype] = i
                    
                # Stop once we have all examples
                if all(v is not None for v in examples.values()):
                    break
        
        return examples
    
    def _create_statistics_plot(self) -> None:
        """Create frame statistics visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Frame Statistics Analysis", fontsize=16)
        
        # Collect statistics
        frame_sizes = {"I": [], "P": [], "B": []}
        for frame in self.frames:
            frame_sizes[frame["pict_type"]].append(int(frame["pkt_size"]))
        
        # Frame size distribution
        ax1 = axes[0, 0]
        data = [frame_sizes["I"], frame_sizes["P"], frame_sizes["B"]]
        bp = ax1.boxplot(data, labels=["I", "P", "B"])
        ax1.set_title("Frame Size Distribution")
        ax1.set_ylabel("Size (bytes)")
        ax1.set_xlabel("Frame Type")
        ax1.grid(True, alpha=0.3)
        
        # Average sizes
        ax2 = axes[0, 1]
        avg_sizes = [np.mean(sizes) if sizes else 0 for sizes in frame_sizes.values()]
        bars = ax2.bar(["I", "P", "B"], avg_sizes, color=['red', 'green', 'blue'])
        ax2.set_title("Average Frame Size")
        ax2.set_ylabel("Size (bytes)")
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, avg_sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{int(val)}', ha='center', va='bottom')
        
        # Frame type distribution
        ax3 = axes[1, 0]
        counts = [len(frame_sizes[t]) for t in ["I", "P", "B"]]
        ax3.pie(counts, labels=["I", "P", "B"], autopct='%1.1f%%',
                colors=['red', 'green', 'blue'])
        ax3.set_title("Frame Type Distribution")
        
        # Timeline
        ax4 = axes[1, 1]
        indices = list(range(len(self.frames)))
        colors = {'I': 'red', 'P': 'green', 'B': 'blue'}
        for ftype in ['I', 'P', 'B']:
            idx = [i for i, f in enumerate(self.frames) if f['pict_type'] == ftype]
            sizes = [int(self.frames[i]['pkt_size']) for i in idx]
            ax4.scatter(idx, sizes, c=colors[ftype], label=ftype, alpha=0.6, s=20)
        ax4.set_title("Frame Size Timeline")
        ax4.set_xlabel("Frame Index")
        ax4.set_ylabel("Size (bytes)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "frame_statistics.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def cleanup_intermediate_files(self) -> None:
        """Remove intermediate files to save space."""
        logger.info("Cleaning up intermediate files...")
        
        # Keep only the visualization outputs
        dirs_to_clean = ["frames", "motion_vectors", "differences", "yuv_analysis"]
        for dir_name in dirs_to_clean:
            dir_path = self.output_dir / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)


def main():
    """Main entry point for frame residual analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze video frame residuals and motion vectors")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep intermediate files")
    
    args = parser.parse_args()
    
    # Use default video if not specified
    if not args.video:
        video_files = list(Path("data").glob("*.mp4"))
        if video_files:
            args.video = str(video_files[0])
        else:
            print("No video files found in data/ directory")
            return
    
    # Run analysis
    analyzer = FrameResidualAnalyzer(args.video, args.output)
    analyzer.create_comprehensive_visualization()
    
    if not args.no_cleanup:
        analyzer.cleanup_intermediate_files()
    
    print(f"\nAnalysis complete! Generated files:")
    print(f"- {args.output}/frame_residual_analysis.png")
    print(f"- {args.output}/frame_statistics.png")


if __name__ == "__main__":
    main()
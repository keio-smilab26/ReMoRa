# mviz

Extract and visualize H.264 codec motion vectors from video files. `mviz`
is shipped as a subpackage of the parent ReMoRa repo and is installed
together with it (`pip install -e .` or `uv sync` from the repo root).

## Quick Start

### Extract all features (codec metadata + motion vectors + frames)
```bash
python -m mviz.extract_features video.mp4 --mode all --output output/
```

### Extract with motion vector overlay video
```bash
python -m mviz.extract_features video.mp4 --mode motion --output output/ --visualize
```

This generates a static residual grid image and a motion vector overlay video (using FFmpeg's `codecview` filter) under `output/visualizations/`.

### Extract codec motion vectors only
```bash
python -m mviz.extractors.codec_mvs video.mp4
```

### Visualize motion vectors + residuals overlaid on frames
```bash
python -m mviz.visualizers.residual_overlay video.mp4
```

### Extract codec-level features (frame types, packet sizes, quality)
```bash
python -m mviz.extractors.codec_features video.mp4
```

## Directory Structure

```
mviz/
├── __init__.py
├── extract_features.py                # Main CLI entry point
├── extract_mvs_ffmpeg                 # Compiled binary for raw MV extraction
├── extractors/                        # Motion vector & codec feature extraction
│   ├── codec_mvs.py                   # Core codec MV extractor (uses extract_mvs_ffmpeg)
│   ├── motion_vectors.py              # FFmpeg-based MV extraction
│   ├── codec_features.py              # Codec metadata (frame types, packet sizes)
│   └── frame_selector.py              # Keyframe / I-frame selection strategies
├── visualizers/                       # Overlay & qualitative visualization
│   ├── gop_visualization.py           # GOP-aware MV arrow overlays on frames
│   ├── residual_grid.py               # Multi-panel MV + residual grid
│   ├── residual_overlay.py            # MV + residual overlay on decoded frames
│   └── visualization_utils.py         # Shared visualization utilities
├── utils/                             # Supporting utilities
│   ├── video_encoder.py               # H.264 re-encoding with controlled GOP
│   └── frame_residuals.py             # Frame residual computation
└── examples/                          # Usage examples
    └── ml_features.py                 # Preparing MV features for ML models
```

## Requirements

- Python 3.11+
- FFmpeg (with libavcodec)
- `extract_mvs_ffmpeg` binary — a prebuilt **Linux x86-64** copy is shipped
  alongside this README. On other platforms, build it from the upstream
  FFmpeg example (`doc/examples/extract_mvs.c`):

  ```bash
  git clone https://git.ffmpeg.org/ffmpeg.git
  cd ffmpeg && ./configure && make
  gcc doc/examples/extract_mvs.c -o extract_mvs_ffmpeg \
      $(pkg-config --cflags --libs libavformat libavcodec libavutil)
  ```

  Drop the resulting binary into `mviz/` (or anywhere on `$PATH`).

## License

Distributed under the same BSD-3-Clause-Clear license as the parent
ReMoRa repository. See the top-level [`LICENSE`](../LICENSE).

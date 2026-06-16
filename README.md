# 🌟 [CVPR26] ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding
[![Conference](https://img.shields.io/badge/CVPR-2026-bbeaff.svg)](https://arxiv.org/abs/2602.16412)
[![arXiv](https://img.shields.io/badge/arXiv-2602.16412-b31b1b.svg)](https://arxiv.org/abs/2602.16412)

- Accepted at CVPR 2026
- 🌐 [project page](https://remora-v1rcm.kinsta.page/)
- 📄 [arXiv](https://arxiv.org/abs/2602.16412)

#### Authors

Daichi Yashima<sup>1,3</sup>&nbsp;&nbsp;&nbsp;&nbsp;Shuhei Kurita<sup>2,3</sup>&nbsp;&nbsp;&nbsp;&nbsp;Yusuke Oda<sup>3</sup>&nbsp;&nbsp;&nbsp;&nbsp;Komei Sugiura<sup>1</sup>

<sup>1</sup>Keio University&nbsp;&nbsp;&nbsp;&nbsp;<sup>2</sup>NII&nbsp;&nbsp;&nbsp;&nbsp;<sup>3</sup>NII LLMC

## Installation

```bash
uv sync --extra train

# Or with pip
pip install -e ".[train]"
```

For the motion-vector extraction / visualization utilities under
[`mviz/`](mviz/README.md), add the `mviz` extra (covered by `train` as
well):

```bash
pip install -e ".[mviz]"
```

## Checkpoint

The pretrained ReMoRa checkpoint is available on Hugging Face:

- 🤗 [naisekizero/ReMoRa](https://huggingface.co/naisekizero/ReMoRa)

## Inference

```bash
python infer_with_mv.py \
    --checkpoint checkpoints/ReMoRa-7B \
    --base lmms-lab/LLaVA-Video-7B-Qwen2 \
    --video /path/to/video.mp4 \
    --prompt "Describe what happens in this video."
```

## Extracting motion vectors for training / batch eval

```bash
python scripts/extract_motion_vectors.py \
    --video-root /path/to/your/videos \
    --output-dir DATAS/motion_vectors \
    --fps 16 --block-size 16
```

```bash
# Training
bash scripts/train_remora.sh  # add --motion_vector_dir DATAS/motion_vectors
# or:
export REMORA_MV_DIR=DATAS/motion_vectors

# Batch evaluation
python llava/eval/infer.py \
    --motion_vector_dir DATAS/motion_vectors \
    ...
```

## Acknowledgements

This codebase builds on:

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [BIMBA](https://github.com/SonyResearch/BIMBA)
- [mamba](https://github.com/state-spaces/mamba)

## License

This work is licensed under the BSD-3-Clause-Clear license. See [LICENSE](LICENSE).

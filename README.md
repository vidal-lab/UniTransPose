<div align="center">

<h1>🕺 UniTransPose</h1>

<h3>Transformers with Joint Tokens and Local-Global Attention<br/>for Efficient Human Pose Estimation</h3>

**[Kaleab A. Kinfu](https://kaleab.me/)** &nbsp;·&nbsp; **[René Vidal](https://www.grasp.upenn.edu/people/rene-vidal/)**

*Center for Innovation in Data Engineering and Science, University of Pennsylvania*

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TPAMI%202026-00629B.svg?logo=ieee)](https://www.computer.org/csdl/journal/tp/5555/01/11554363/2h9GijE4F7G)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Built on MMPose](https://img.shields.io/badge/Built%20on-MMPose%201.3-1f7fbf.svg)](https://github.com/open-mmlab/mmpose)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)

<p align="center">
  <img src="assets/architecture.png" width="96%" alt="UniTransPose architecture"/>
</p>

</div>

> **UniTransPose** is a multi-scale vision transformer for 2D human pose estimation. It pairs an efficient encoder that mixes **local and global attention** through learnable **joint tokens** (Joint-Aware Global-Local attention) with an efficient **sub-pixel CNN decoder**, and is trained on a **unified 23-keypoint skeleton** that merges six benchmarks so a single model learns from all of them at once.

This is the official implementation, released as an [MMPose](https://github.com/open-mmlab/mmpose) project so it plugs directly into MMPose's training, evaluation, and inference framework.

## 📑 Table of Contents

- [📰 News](#-news)
- [😮 Highlights](#-highlights)
- [📖 Abstract](#-abstract)
- [🚀 Main Results](#-main-results)
- [🛠️ Requirements and Installation](#%EF%B8%8F-requirements-and-installation)
- [🐳 Model Zoo](#-model-zoo)
- [🤖 Usage](#-usage)
- [🗝️ Training & Validating](#%EF%B8%8F-training--validating)
- [🔒 License](#-license)
- [✏️ Citation](#%EF%B8%8F-citation)

## 📰 News

- **[2026]** 🎉 UniTransPose is accepted to **IEEE TPAMI** — [read the paper](https://www.computer.org/csdl/journal/tp/5555/01/11554363/2h9GijE4F7G).
- **[2023]** 📌 The efficiency-focused companion model **EViTPose** was presented at **BMVC 2023 (Oral)**.

## 😮 Highlights

UniTransPose targets the three desiderata of pose estimation — **accuracy, efficiency, and robustness** — in a single model.

<details open>
<summary><b>💡 Joint-Aware Global-Local (JAGL) attention</b></summary>
<br/>
A small set of learnable <b>joint tokens</b> gathers global context from the patch tokens
(<i>patch-to-joint</i>), mixes information among themselves (<i>joint-to-joint</i>), and scatters
the context back onto the patch tokens (<i>joint-to-patch</i>). Because the joints are far fewer
than the patches, this injects global reasoning into windowed local attention at
<b>linear cost</b> in the number of patches — unlike the quadratic cost of full self-attention.
</details>

<details open>
<summary><b>⚡ Efficient sub-pixel CNN decoder</b></summary>
<br/>
A PixelShuffle-based head upsamples in the <b>low-resolution</b> feature space and then refines,
producing accurate Gaussian heatmaps with far fewer operations than transpose-convolution
decoders.
</details>

<details open>
<summary><b>🦴 Unified 23-keypoint skeleton</b></summary>
<br/>
Joints from MS-COCO, MPII, AI Challenger, CrowdPose, JRDB-Pose, and OCHuman are merged into a
single skeleton, enabling robust <b>multi-dataset</b> training, zero-shot transfer to shared
joints, and flexible joint selection at inference.
</details>

<details open>
<summary><b>🔀 Hybrid decoding</b></summary>
<br/>
A heatmap decoder (accuracy) and a keypoint-regression head (speed) share the same encoder —
choose either at inference time depending on your accuracy/latency budget.
</details>

## 📖 Abstract

<details>
<summary>Click to expand</summary>
<br/>
Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) have led to significant
progress in 2D body pose estimation, yet striking a good balance between accuracy, efficiency,
and robustness remains a challenge: CNNs are efficient but struggle with long-range
dependencies, while ViTs capture them but suffer from quadratic complexity. We propose two
ViT-based models for accurate, efficient, and robust 2D pose estimation. <b>EViTPose</b> operates
efficiently without sacrificing accuracy by using learnable joint tokens to select and process a
subset of the most important body patches, giving direct control over the accuracy/efficiency
trade-off. <b>UniTransPose</b> efficiently handles multiple scales by combining an efficient
multi-scale transformer encoder that uses both local and global attention with an efficient
sub-pixel CNN decoder for better speed and accuracy. By incorporating all joints from different
benchmarks into a unified skeletal representation, we train robust models that learn from
multiple datasets simultaneously and perform well across a wide range of scenarios — including
pose variations, lighting conditions, and occlusions. EViTPose reduces computational complexity
by 30–44% in GFLOPs with a minimal accuracy drop (0–3.5%), and UniTransPose improves accuracy by
0.9–43.8% across six benchmarks.
</details>

## 🚀 Main Results

Trained on the unified skeleton, **UniTransPose improves over the prior state of the art** across six
2D pose benchmarks:

| Benchmark | Metric | Improvement over SOTA |
|:--|:--:|:--:|
| MS-COCO | AP | **+0.9 – 2.4** |
| AI Challenger | AP | **+3.3 – 5.7** |
| OCHuman | AP | **+6.2 – 43.8** |
| MPII | PCKh | ✅ consistent gains |
| CrowdPose | AP | ✅ consistent gains |
| JRDB-Pose | AP | ✅ consistent gains |

<p align="center">
  <img src="assets/result.jpg" width="58%" alt="Qualitative result"/>
</p>

## 🛠️ Requirements and Installation

- Python ≥ 3.8, PyTorch ≥ 2.0 (CUDA build matching your GPU)
- [MMEngine](https://github.com/open-mmlab/mmengine), [MMCV](https://github.com/open-mmlab/mmcv) (`>=2.0.1, <2.2.0`), [MMDetection](https://github.com/open-mmlab/mmdetection) (`>=3.0`) for top-down person detection
- `timm`, `einops`

```bash
# 1) environment
conda create -n unitranspose python=3.9 -y && conda activate unitranspose
pip install torch torchvision                       # pick the CUDA build for your machine

# 2) OpenMMLab stack
pip install -U openmim
mim install "mmcv>=2.0.1,<2.2.0" "mmengine" "mmdet>=3.0"

# 3) this repository (an MMPose fork) + extras
pip install -e .
pip install timm einops
```

The model lives in [`projects/unitranspose`](projects/unitranspose). Add it to `PYTHONPATH` so the
config's `custom_imports=['unitranspose']` resolves.

## 🐳 Model Zoo

| Model | Input | Joints | Config | Checkpoint |
|:--|:--:|:--:|:--:|:--:|
| UniTransPose (COCO-17) | 256×192 | 17 | [config](projects/unitranspose/unitranspose/configs/unitranspose_coco-256x192.py) | · *release TBA* |
| UniTransPose (Unified) | 256×192 | 23 | [config](projects/unitranspose/unitranspose/configs/unitranspose_multi23-256x192.py) | · *release TBA* |

## 🤖 Usage

**Inference**

```python
from mmpose.apis import init_model, inference_topdown

model = init_model(
    'unitranspose/configs/unitranspose_coco-256x192.py',
    'unitranspose/unitranspose.pth',
    device='cuda:0')

# bboxes: np.ndarray [N, 4] in xyxy (e.g. from an MMDetection person detector)
results = inference_topdown(model, img, bboxes=bboxes_xyxy, bbox_format='xyxy')
keypoints = results[0].pred_instances.keypoints     # (1, K, 2)
```

- `unitranspose_coco-256x192.py` returns the **17 COCO** joints.
- `unitranspose_multi23-256x192.py` returns all **23 unified** joints.

```bash
# quick 23-keypoint demo + visualization on the bundled sample image
cd projects/unitranspose/unitranspose
PYTHONPATH=.. python test_predict23.py
```

## 🔒 License

Released under the [Apache 2.0 License](LICENSE), following MMPose.

## ✏️ Citation

If you find UniTransPose useful for your research, please cite:

```bibtex
@article{kinfu2026unitranspose,
  title   = {Transformers with Joint Tokens and Local-Global Attention for Efficient Human Pose Estimation},
  author  = {Kinfu, Kaleab A. and Vidal, Ren\'e},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2026}
}
```

Please also consider citing MMPose:

```bibtex
@misc{mmpose2020,
  title        = {OpenMMLab Pose Estimation Toolbox and Benchmark},
  author       = {MMPose Contributors},
  howpublished = {\url{https://github.com/open-mmlab/mmpose}},
  year         = {2020}
}
```


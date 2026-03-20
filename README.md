# MultihopSpatial: Multi-hop Compositional Spatial Reasoning Benchmark for Vision-Language Models

<p align="center">
  <img src="static/figures/teaser_2.png" width="100%" alt="MultihopSpatial Benchmark Overview">
</p>

<p align="center">
  <a href="https://youngwanlee.github.io/multihopspatial"><b>Project Page</b></a> |
  <a href="https://arxiv.org/abs/2603.18892"><b>Paper</b></a> |
  <a href="https://huggingface.co/etri-vilab/MultiHopSpatial-Qwen3-VL-4B-Instruct"><b>Model</b></a>
</p>

## Overview

**MultihopSpatial** is a benchmark designed to evaluate whether vision-language models (VLMs) demonstrate robustness in **multi-hop compositional spatial reasoning**. Unlike existing benchmarks that only assess single-step spatial relations, MultihopSpatial features queries with **1 to 3 reasoning hops** paired with **visual grounding evaluation**, exposing a critical blind spot: models achieving high multiple-choice accuracy often lack proper spatial localization.

All 4,500 benchmark QA pairs and bounding boxes are **strictly annotated by ten trained human experts** with an inter-rater agreement of 90% (Krippendorff's α = 0.90).

## Key Features

- **Multi-hop Composition**: Tests 1-hop, 2-hop, and 3-hop sequential spatial reasoning, mirroring real-world embodied AI needs.
- **Grounded Evaluation**: Addresses the "lucky guess" problem — models must both select the correct answer AND localize it via bounding box (Acc@50IoU).
- **Perspective-taking**: Includes both ego-centric and exo-centric viewpoints.
- **Three Spatial Categories**: Attribute (ATT), Position (POS), and Relation (REL), composable into multi-hop questions.
- **Training Data**: MultihopSpatial-Train (6,791 samples) supports post-training via reinforcement learning (e.g., GRPO).

## Evaluation
Comming soon


## Citation
```bibtex
@article{lee2025multihopspatial,
  title={MultihopSpatial: Multi-hop Compositional Spatial Reasoning Benchmark for Vision-Language Models},
  author={Lee, Youngwan and Jang, Soojin and Cho, Yoorhim and Lee, Seunghwan and Lee, Yong-Ju and Hwang, Sung Ju},
  journal={arXiv preprint arXiv:2603.18892},
  year={2025}
}
```
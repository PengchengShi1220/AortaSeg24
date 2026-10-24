# [AortaSeg24 Challenge](https://aortaseg24.grand-challenge.org/)

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow.svg)](https://huggingface.co/spc819/AortaSeg24)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-b31b1b.svg)](https://arxiv.org/abs/2511.14187)

### 📰 News
- **October 19, 2026**: 🤗 Pretrained weights of the final-phase solution (Dataset824 `3d_fullres` folds 0–4 + `all`, Dataset825 `3d_lowres` fold 0 + `all`) are released on [Hugging Face: spc819/AortaSeg24](https://huggingface.co/spc819/AortaSeg24).
- **October 30, 2024**: 🥈 Achieved **2nd place** in the Final Test Phase! [View official rankings](https://grand-challenge.org/forums/forum/multi-class-segmentation-of-aortic-branches-and-zones-in-cta-726/topic/challenge-final-rankings-2826/)
- **October 10, 2024**: The complete code for the [AortaSeg24 Challenge](https://aortaseg24.grand-challenge.org/) Final Test Phase is now available.
## Pretrained Weights

The trained checkpoints of the final-phase solution are released on Hugging Face:
**[spc819/AortaSeg24](https://huggingface.co/spc819/AortaSeg24)**.

The **model weights are licensed under CC BY-NC-SA 4.0 (non-commercial)**; the code in this
repository is Apache-2.0. See [License](#license).

The two models used by `inference.py` correspond to the following folders:

| Path under `nnUNet_results/` | Stage | Configuration | Classes | Folds released |
|---|---|---|---|---|
| `Dataset825_AortaSeg24_CTA_bin_50/nnUNetTrainer_CE_DC_CBDC_NoMirroring_500epochs__nnUNetPlans__3d_lowres` | 1 — aorta localization | `3d_lowres` | 2 (aorta + background) | `fold_0`, `fold_all` |
| `Dataset824_AortaSeg24_CTA_50/nnUNetTrainer_Hierarchical_CBDC_AortaSeg23_NoMirroring_500epochs__nnUNetPlans__3d_fullres` | 2 — multi-class segmentation | `3d_fullres` | 24 (23 aortic segments/zones + background) | `fold_0`–`fold_4`, `fold_all` |

Each trainer folder ships `plans.json`, `dataset.json` and `dataset_fingerprint.json`, and each
`fold_*` subfolder contains its `checkpoint_final.pth` — everything the predictor needs.
Training logs and curve images are not included.

`inference.py` loads exactly these two folders and uses `use_folds=(0, 'all')` for the
low-resolution stage and `use_folds=(0, 1, 2, 3, 4, 'all')` for the full-resolution stage,
so the released checkpoints cover the inference configuration as-is.

### Download

```bash
hf download spc819/AortaSeg24 --repo-type model --local-dir $nnUNet_results
```

The repository mirrors the `nnUNet_results` layout, so the `Dataset824_*` / `Dataset825_*`
folders land exactly where `inference.py` expects them (`/opt/app/nnUNet/nnUNet_results`
inside the container).

With the weights in place, skip [Step 2](#step-2-model-training) and go straight to
[Step 3: Inference](#step-3-inference).

### Use with the code or the Docker build

- **nnU-Net / Python**: point `nnUNet_results` (or the path hard-coded in `inference.py`)
  at the download location and run `test_run.sh`.
- **Docker**: the `Dockerfile` copies `Dataset824_AortaSeg24_CTA_50` and
  `Dataset825_AortaSeg24_CTA_bin_50` from the build context into
  `/opt/app/nnUNet/nnUNet_results/`, so the downloaded folders can be placed in the build
  context next to the code and built directly.

## Dependencies

This project makes use of the following main resources. Please refer to their GitHub repositories for more details:

- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
- [fractal softmax](https://github.com/PengchengShi1220/fractal-softmax)
- [cbDice](https://github.com/PengchengShi1220/cbDice)

Please note that the hierarchical semantic loss will be added in the [fractal softmax](https://github.com/PengchengShi1220/fractal-softmax).

## Installation and Usage Guide

### Step 1: Installation
To begin, extract the provided archive `nnUNet_hierarchical_cbdc_nnUNet_v2.5.zip` by executing the following command:
```
unzip nnUNet_hierarchical_cbdc_nnUNet_v2.5.zip
```
Next, navigate to the extracted directory and install the package in editable mode:
```
cd nnUNet_hierarchical_cbdc_nnUNet_v2.5
pip install -e .
```

### Step 2: Model Training
- **Low-Resolution Binary Segmentation**: To train the low-resolution binary segmentation model, use the dataset `Dataset825_AortaSeg24_CTA_bin_50`, which contains two classes (including the background). Execute the following command to initiate training:
```
bash task_run_3d_lowres_825.sh
```
- **Full-Resolution Multi-Class Segmentation**: Upon completion of the low-resolution training phase, proceed to train the full-resolution multi-class segmentation model using the dataset `Dataset824_AortaSeg24_CTA_50`, which contains 24 classes (including the background). Execute:
```
bash task_run_3d_fullres_824.sh
```

### Step 3: Inference
To perform inference, execute the following script:
```
bash test_run.sh
```

The inference process, as defined in `inference.py`, employs a two-stage methodology:
1. **Low-Resolution Binary Segmentation**: Initially, a low-resolution 3D model conducts binary segmentation of the entire aorta, identifying all foreground vessels. Following this, a region of interest (ROI) is extracted using a scaling factor (*m*) of 2.0, thereby extending the ROI for more detailed analysis.
2. **Full-Resolution Multi-Class Segmentation**: Within the extracted ROI, a full-resolution 3D model performs multi-class segmentation. The final segmentation output is obtained by integrating the segmented ROI back into the original image dimensions.

This hierarchical approach significantly enhances inference efficiency, reducing runtime by over five-fold compared to conventional single-stage methods.

### Step 4: Saving the Docker Container
To save your Docker container following inference, execute the following command:
```
bash save.sh
```

## Citation
```
@misc{shi2025hierarchicalsemanticlearningmulticlass,
      title={Hierarchical Semantic Learning for Multi-Class Aorta Segmentation}, 
      author={Pengcheng Shi},
      year={2025},
      eprint={2511.14187},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.14187}, 
}
```
```
@inproceedings{shi2024centerline,
  title={Centerline Boundary Dice Loss for Vascular Segmentation},
  author={Shi, Pengcheng and Hu, Jiesi and Yang, Yanwu and Gao, Zilve and Liu, Wei and Ma, Ting},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={46--56},
  year={2024},
  organization={Springer}
}
```
```
@article{isensee2021nnu,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## License

- **Code** (this repository): **Apache-2.0** — see [LICENSE](LICENSE).
- **Model weights** ([Hugging Face](https://huggingface.co/spc819/AortaSeg24)):
  **CC BY-NC-SA 4.0 (non-commercial)**; attribution required, derivatives under the same
  license. See <https://creativecommons.org/licenses/by-nc-sa/4.0/>.

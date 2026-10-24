---
pipeline_tag: image-segmentation
license: cc-by-nc-sa-4.0
---

# AortaSeg24: Hierarchical Semantic Learning for Multi-Class Aorta Segmentation

[![Paper](https://img.shields.io/badge/Paper-Arxiv-b31b1b.svg)](https://arxiv.org/abs/2511.14187)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow.svg)](https://huggingface.co/spc819/AortaSeg24)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue.svg?logo=github)](https://github.com/PengchengShi1220/AortaSeg24)
[![Challenge](https://img.shields.io/badge/AortaSeg24-grand--challenge-4CAF50.svg)](https://aortaseg24.grand-challenge.org/)

This repository hosts the **trained nnU-Net checkpoints** of our solution to the
[AortaSeg24 Challenge](https://aortaseg24.grand-challenge.org/), which achieved
**🥈 2nd place in the Final Test Phase** ([official rankings](https://grand-challenge.org/forums/forum/multi-class-segmentation-of-aortic-branches-and-zones-in-cta-726/topic/challenge-final-rankings-2826/)).

Training and inference code (including the customized nnU-Net fork, the hierarchical
semantic loss and the cbDice loss) is at
**[github.com/PengchengShi1220/AortaSeg24](https://github.com/PengchengShi1220/AortaSeg24)**.

## Repository contents

The layout mirrors `nnUNet_results/` exactly, so the download can be dropped in as-is:

```
Dataset824_AortaSeg24_CTA_50/
└── nnUNetTrainer_Hierarchical_CBDC_AortaSeg23_NoMirroring_500epochs__nnUNetPlans__3d_fullres/
    ├── dataset.json  plans.json  dataset_fingerprint.json
    └── fold_{0,1,2,3,4,all}/
        ├── checkpoint_final.pth      # 24-class full-resolution model (23 foreground classes + background)
        └── debug.json
Dataset825_AortaSeg24_CTA_bin_50/
└── nnUNetTrainer_CE_DC_CBDC_NoMirroring_500epochs__nnUNetPlans__3d_lowres/
    ├── dataset.json  plans.json  dataset_fingerprint.json
    └── fold_{0,all}/
        ├── checkpoint_final.pth      # 2-class low-resolution model (aorta vs. background)
        └── debug.json
```

- **Dataset824_AortaSeg24_CTA_50** — full-resolution multi-class model (`3d_fullres`), 5 folds + `fold_all`.
- **Dataset825_AortaSeg24_CTA_bin_50** — low-resolution binary model (`3d_lowres`), fold 0 + `fold_all`, used to localize the aorta before the multi-class stage.
- `params.json` — model metadata (framework, stage/dataset mapping, folds, class counts).
- Only weights and JSON configs are published here; training logs and curve images are not included.

Both models were trained for 500 epochs with mirroring disabled
(`*_NoMirroring_500epochs`).

## Method

A **hierarchical two-stage pipeline**:

1. **Low-resolution binary segmentation** — a `3d_lowres` nnU-Net segments the whole
   aorta (all foreground vessels) on downsampled volumes.
2. **ROI extraction** — the binary prediction is expanded by a scaling factor *m = 2.0*
   to obtain a generous region of interest.
3. **Full-resolution multi-class segmentation** — a `3d_fullres` nnU-Net predicts the
   24 classes (23 aortic segments/zones + background) inside the ROI, and the result is
   pasted back into the original image grid.

This hierarchical design reduces inference time by **more than five-fold** compared with
a single-stage full-volume multi-class model, while keeping full-resolution accuracy
inside the ROI.

The multi-class model is trained with **hierarchical semantic learning** through
[fractal softmax](https://github.com/PengchengShi1220/fractal-softmax) and the
[cbDice](https://github.com/PengchengShi1220/cbDice) (centerline boundary Dice) loss.
See the paper for details.

## Requirements

**Python Version:** 3.10 (same environment as the code repository)

### Installation

```bash
unzip nnUNet_hierarchical_cbdc_nnUNet_v2.5.zip      # from the GitHub repository
cd nnUNet_hierarchical_cbdc_nnUNet_v2.5
pip install -e .
```

## Quick start with these checkpoints

1. **Download the weights into `nnUNet_results/`** (the dataset folders must land directly
   inside it):

```bash
hf download spc819/AortaSeg24 --repo-type model --local-dir /path/to/nnUNet_results
```

2. **Set the nnU-Net environment variables** (as usual for nnU-Net):

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

3. **Run the two-stage inference** with the code repository:

```bash
bash test_run.sh        # edit the paths / fold inside before running
```

The inference entry point `inference.py` runs the low-resolution binary model, crops the
expanded ROI and then applies the full-resolution multi-class model.

## Training guidance

Training is also two-stage; run the scripts from the GitHub repository on the
corresponding datasets:

```bash
bash task_run_3d_lowres_825.sh     # 1) Dataset825_AortaSeg24_CTA_bin_50  (2 classes)
bash task_run_3d_fullres_824.sh    # 2) Dataset824_AortaSeg24_CTA_50      (24 classes)
```

Prepare the datasets in the standard nnU-Net format (`imagesTr/`, `labelsTr/`,
`dataset.json`) and preprocess them with `nnUNetv2_plan_and_preprocess` before training.
The hierarchical semantic loss and cbDice are provided by the customized nnU-Net fork
installed in the step above.

## Inference guidance

```bash
bash test_run.sh                   # two-stage inference
bash save.sh                       # optionally package the container for submission
```

- Input: CTA volumes in NIfTI format (`*_0000.nii.gz`, standard nnU-Net naming).
- Output: label maps with 0 = background and 1–23 = aortic segments/zones; the exact class
  definition follows the [AortaSeg24 challenge description](https://aortaseg24.grand-challenge.org/).
- The ROI scaling factor (*m = 2.0*) and the fold selection are set in the inference
  script; the checkpoints published here cover folds 0–4 and `fold_all` for the
  multi-class model, and fold 0 / `fold_all` for the binary model.

## License and citation

- **Code** ([github.com/PengchengShi1220/AortaSeg24](https://github.com/PengchengShi1220/AortaSeg24)):
  **Apache-2.0**.
- **Model weights** (this repository): **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
  (non-commercial)**; attribution required, derivatives under the same license.

## Citation

```
@incollection{shi2024hierarchical,
  title={Hierarchical Semantic Learning for Multi-class Aorta Segmentation},
  author={Shi, Pengcheng},
  booktitle={International Challenge on Multi-class Segmentation of the Aorta},
  pages={12--22},
  year={2024},
  publisher={Springer}
}
```

If you use the cbDice loss, please also cite:

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

## Acknowledgements

This solution is built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet),
[fractal softmax](https://github.com/PengchengShi1220/fractal-softmax) and
[cbDice](https://github.com/PengchengShi1220/cbDice). We thank the AortaSeg24 organizers
and the grand-challenge.org team for the dataset and the evaluation platform.

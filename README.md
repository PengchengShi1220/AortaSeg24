# [AortaSeg24 Challenge](https://aortaseg24.grand-challenge.org/)

## Dependencies

This project makes use of the following main resources. Please refer to their GitHub repositories for more details:

- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
- [NexToU](https://github.com/PengchengShi1220/NexToU)
- [cbDice](https://github.com/PengchengShi1220/cbDice)

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
@article{shi2023nextou,
  title={NexToU: Efficient Topology-Aware U-Net for Medical Image Segmentation},
  author={Shi, Pengcheng and Guo, Xutao and Yang, Yanwu and Ye, Chenfei and Ma, Ting},
  journal={arXiv preprint arXiv:2305.15911},
  year={2023}
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

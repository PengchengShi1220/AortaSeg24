#!/bin/bash
nnUNetv2_plan_and_preprocess -d 825 -c "3d_lowres"

# nnUNetv2:
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset825_AortaSeg24_CTA_bin_50 3d_lowres 0 -tr nnUNetTrainer_CE_DC_CBDC_NoMirroring_500epochs --c
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset825_AortaSeg24_CTA_bin_50 3d_lowres all -tr nnUNetTrainer_CE_DC_CBDC_NoMirroring_500epochs --c

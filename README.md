# MICCAI 2026 Submission

This repository contains the implementation code for our MICCAI 2026 submission.

> ### :lock: Anonymous Review Notice
> This repository is anonymized for double-blind review.  
> Author information and license will be added upon acceptance.  
> Full code and training scripts will be released after the paper is accepted.


## Description

We propose a novel framework for semi-supervised medical image segmentation using chained diffusion-based pseudo-label refinement.

## Requirements

- Python 3.8+
- PyTorch >= 1.10
- CUDA 11.x
- Other dependencies are listed in `requirements.txt` (to be provided if necessary)

## Data Preparation

- **ACDC**: Download from the official SSL4MIS repository: [https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC)

- **MS-CMRSEG19**: Download from the official challenge website: [https://zmiclab.github.io/zxh/0/mscmrseg19/](https://zmiclab.github.io/zxh/0/mscmrseg19/)

- **Task05 Prostate (Medical Segmentation Decathlon)**: [http://medicaldecathlon.com/](http://medicaldecathlon.com/)


## Usage

Train the model with the following command:

# Example: Train the segmentation model on ACDC with 1 labeled patient
CUDA_VISIBLE_DEVICES=0 python train_diff_chain_ACDC.py \
    --exp ACDC/diffchain \
    --labelnum 1 \
    --num_classes 4 \
    --root_path ./datasets/ACDC

## Acknowledgement

We sincerely appreciate the following open-source projects for their valuable contributions, which our work builds upon:

- [DiffRect](https://github.com/CUHK-AIM-Group/DiffRect)
- [U-KAN](https://github.com/CUHK-AIM-Group/U-KAN)
- [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)
- [guided-diffusion](https://github.com/openai/guided-diffusion)
- [GSS](https://github.com/fudan-zvg/GSS)
- [DiffUNet](https://github.com/ge-xing/Diff-UNet)

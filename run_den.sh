#!/bin/bash
source ../../anaconda3/etc/profile.d/conda.sh
conda activate CSCNet

python train.py --n_ch 1 --deg denoising --noise_level 15 --num_epochs 500 --model_tag 提案手法
python train.py --n_ch 1 --deg denoising --noise_level 25 --num_epochs 500 --model_tag 提案手法
python train.py --n_ch 1 --deg denoising --noise_level 50 --num_epochs 500 --model_tag 提案手法

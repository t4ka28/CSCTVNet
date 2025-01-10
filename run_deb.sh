#!/bin/bash
source ../../anaconda3/etc/profile.d/conda.sh
conda activate CSCNet

python train.py --n_ch 1 --deg deblurring --blur_level 3 --noise_level 0.1 --num_epochs 500 --model_tag 提案手法
python train.py --n_ch 1 --deg deblurring --blur_level 3 --noise_level 0.2 --num_epochs 500 --model_tag 提案手法
python train.py --n_ch 1 --deg deblurring --blur_level 5 --noise_level 0.05 --num_epochs 500 --model_tag 提案手法
python train.py --n_ch 1 --deg deblurring --blur_level 5 --noise_level 0.1 --num_epochs 500 --model_tag 提案手法

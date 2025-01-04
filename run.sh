#!/bin/bash
source ../../anaconda3/etc/profile.d/conda.sh
conda activate Myresearch

python train.py --n_ch 3 --deg inpainting --missing_rate 10 --noise_level 0 --num_epochs 500 --model_tag 提案手法
python train.py --n_ch 3 --deg inpainting --missing_rate 30 --noise_level 0 --num_epochs 500 --model_tag 提案手法
python train.py --n_ch 3 --deg inpainting --missing_rate 50 --noise_level 0 --num_epochs 500 --model_tag 提案手法

#!/bin/bash
source ../../anaconda3/etc/profile.d/conda.sh
conda activate CSCNet

# python train.py --n_ch 3 --deg inpainting --missing_rate 10 --noise_level 0 --num_epochs 500 --weight 0.2 --model_tag 提案手法2
python train.py --n_ch 3 --deg inpainting --missing_rate 30 --noise_level 0 --num_epochs 300 --weight 0.2 --model_tag 提案手法2
python train.py --n_ch 3 --deg inpainting --missing_rate 50 --noise_level 0 --num_epochs 300 --weight 0.2 --model_tag 提案手法2
python train.py --n_ch 3 --deg inpainting --missing_rate 70 --noise_level 0 --num_epochs 300 --weight 0.2 --model_tag 提案手法2

# python test.py --n_ch 3 --deg inpainting --missing_rate 10 --noise_level 0 --model_tag 提案手法2
# python test.py --n_ch 3 --deg inpainting --missing_rate 30 --noise_level 0 --model_tag 提案手法2
# python test.py --n_ch 3 --deg inpainting --missing_rate 50 --noise_level 0 --model_tag 提案手法2
# python test.py --n_ch 3 --deg inpainting --missing_rate 70 --noise_level 0 --model_tag 提案手法2

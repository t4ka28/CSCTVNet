#!/bin/bash
source ../../anaconda3/etc/profile.d/conda.sh
conda activate CSCNet
python test.py --n_ch 1 --blur_level 3 --noise_level 5e-2 --model_tag Box制約なし
python test.py --n_ch 1 --blur_level 3 --noise_level 1e-1 --model_tag Box制約なし
python test.py --n_ch 1 --blur_level 3 --noise_level 2e-1 --model_tag Box制約なし
python test.py --n_ch 1 --blur_level 5 --noise_level 5e-2 --model_tag Box制約なし
python test.py --n_ch 1 --blur_level 5 --noise_level 1e-1 --model_tag Box制約なし
python test.py --n_ch 1 --blur_level 5 --noise_level 2e-1 --model_tag Box制約なし

# python train.py --n_ch 1 --blur_level 3 --noise_level 5e-2 --num_epochs 1000 --model_tag 提案手法フル設定
# python train.py --n_ch 1 --blur_level 3 --noise_level 1e-1 --num_epochs 1000 --model_tag 提案手法フル設定
# python train.py --n_ch 1 --blur_level 3 --noise_level 2e-1 --num_epochs 1000 --model_tag 提案手法フル設定
# python train.py --n_ch 1 --blur_level 5 --noise_level 5e-2 --num_epochs 1000 --model_tag 提案手法フル設定
# python train.py --n_ch 1 --blur_level 5 --noise_level 1e-1 --num_epochs 1000 --model_tag 提案手法フル設定
# python train.py --n_ch 1 --blur_level 5 --noise_level 2e-1 --num_epochs 1000 --model_tag 提案手法フル設定
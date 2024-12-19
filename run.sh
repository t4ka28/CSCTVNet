#!/bin/bash
source ../../anaconda3/etc/profile.d/conda.sh
conda activate MyResearch
python train.py --n_ch 1 --blur_level 3 --noise_level 1e-1 --num_filters 3  --num_epochs 500 --model_tag Ablation_k3
python train.py --n_ch 1 --blur_level 3 --noise_level 1e-1 --num_filters 5  --num_epochs 500 --model_tag Ablation_k5
python train.py --n_ch 1 --blur_level 3 --noise_level 1e-1 --num_filters 7  --num_epochs 500 --model_tag Ablation_k7
python train.py --n_ch 1 --blur_level 3 --noise_level 1e-1 --num_filters 9  --num_epochs 500 --model_tag Ablation_k9
python train.py --n_ch 1 --blur_level 3 --noise_level 1e-1 --num_filters 11 --num_epochs 500 --model_tag Ablation_k11
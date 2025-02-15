import os
import sys
import cv2
import util
import torch
import IPython
import argparse
import dataloaders

import numpy as np
import matplotlib.pyplot as plt

from util import Phi
from os.path import join
from tqdm import tqdm
from torchvision import transforms

from models import CSCTV, CSCDVTV
from models import CSCTVParams, CSCDVTVParams

#GPU使えるかどうか
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    #単精度で計算を行う
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print(f'GPU name: {torch.cuda.get_device_name()}')
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    
NONEGAPARAM = ["lam", "lam1", "lam2", "lam3", "gam", "gam1", "gam2", "gam3"]

parser = argparse.ArgumentParser()
# 学習モデルを見つける際に必要な情報
parser.add_argument("--n_ch", type=int, dest="n_ch", help="number of channels", default=3)
parser.add_argument("--deg", type=str, dest="deg", help="The kind of the degradation.", default="debluring")
parser.add_argument("--blur_level", type=int, dest="blur_level", help="Should be an int in the range [0,255]", default=3)
parser.add_argument("--missing_rate", type=int, dest="missing_rate", help="Should be an int in the range [0,100]", default=50)
parser.add_argument("--noise_level", type=float, dest="noise_level", help="Should be an int in the range [0,255]", default=5e-2)
parser.add_argument("--model_tag", type=str, dest="model_tag", help="Deteal of the experiment.", default=None)
parser.add_argument("--data_path", type=str, dest="data_path", help="Path to the dir containing the training and testing datasets.", default="./datasets/CBSD/")
args = parser.parse_args()

###################################################################################
# Setting Parameters
###################################################################################
n_channels = args.n_ch
deg = args.deg
blur_level = args.blur_level
missing_rate = args.missing_rate
noise_std = args.noise_level
model_tag = args.model_tag
data_path = args.data_path
model_name = 'CSCTV' if n_channels == 1 else 'CSCDVTV'

if deg == "denoising":
    print(f"Testing model: channel {n_channels} with noise_std {noise_std} on test images...")
    setting = f"std{noise_std}_{model_tag}"
elif deg == "deblurring":
    print(f"Testing model: channel {n_channels} with blur_level {blur_level} and noise_std {noise_std} on test images...")
    setting = f"blur{blur_level}_std{noise_std}_{model_tag}"
elif deg == "inpainting":
    print(f"Testing model: channel {n_channels} with missing_rate {missing_rate} and noise_std {noise_std} on test images...")
    setting = f"rate{missing_rate}_std{noise_std}_{model_tag}"
else:
    raise ValueError(f"Degtype is not available for use. Got: '{deg}'")

###################################################################################
# Config fileの呼び出し+初期設定
###################################################################################
model_filename = join(model_name, deg, f'{setting}.model')
config_path = join("trained_models", model_name, deg, f'{setting}.config')

with open(config_path) as conf_file:
    conf = conf_file.read()
conf = eval(conf)



K = conf['layer']
n_filters = conf['n_filters']
kernel_size = conf['kernel_size']
DVTV_weight = conf['w']
to_pil = transforms.ToPILImage()
os.makedirs(f"result/{model_name}/{deg}/{setting}/org", exist_ok=True)
os.makedirs(f"result/{model_name}/{deg}/{setting}/deg", exist_ok=True)
os.makedirs(f"result/{model_name}/{deg}/{setting}/output", exist_ok=True)

###################################################################################
# Select Param and Model
###################################################################################
if n_channels == 1:
    model_name = 'CSCTV'
    params = CSCTVParams(deg, n_channels, n_filters, kernel_size, K)
    model = CSCTV(params).cuda()
elif n_channels == 3:
    model_name = 'CSCDVTV'
    params = CSCDVTVParams(deg, n_channels, n_filters, kernel_size, K, DVTV_weight)
    model = CSCDVTV(params).cuda()
else:
    raise ValueError(f"The channel '{n_channels}' is not supported!")
    
model.load_state_dict(torch.load(join("trained_models", model_filename)))

###################################################################################
# Dataset
###################################################################################

test_path = [f'{data_path}/test']
loaders = dataloaders.get_dataloaders([test_path], n_channels, batch_size=1, phase='test')

###################################################################################
# Test
###################################################################################  

psnr = 0
ssim = 0
num_iters = 0

for batch in tqdm(loaders['test']):
    
    phi = Phi(deg, batch, blur_level, missing_rate)
    batch = batch.cuda()
    deg_batch = phi(batch)
    noise = torch.randn_like(batch, device="cuda:0")*noise_std
    z = deg_batch + noise

    # forward
    # track history if only in train
    with torch.set_grad_enabled(False):
        x_hat = model(z, phi, alpha=noise_std)

    org_img, deg_img, output = to_pil(batch[0]), to_pil(util.ProxBoxConstraint(z[0])), to_pil(util.ProxBoxConstraint(x_hat[0]))
    org_img.save(f"result/{model_name}/{deg}/{setting}/org/org{num_iters}.png")
    deg_img.save(f"result/{model_name}/{deg}/{setting}/deg/deg{num_iters}.png")
    output.save(f"result/{model_name}/{deg}/{setting}/output/output{num_iters}.png")

    psnr += cv2.PSNR(util.torch2numpy(batch), util.torch2numpy(x_hat), R=1.0)
    ssim += util.my_ssim(util.torch2numpy(batch), util.torch2numpy(x_hat))

    num_iters += 1

with open(f'result/{model_name}/{deg}/{setting}/Test_log.txt','w') as test_log:
    test_log.write('===========================\n') 
    test_log.write(f'Average PSNR:\t{psnr/num_iters}\n')
    test_log.write(f'Average SSIM:\t{ssim/num_iters}\n')  
    
print('===========================')
print(f'Average PSNR:\t{psnr/num_iters}')
print(f'Average SSIM:\t{ssim/num_iters}')
import os
import cv2
import sys
import util
import uuid
import torch
import IPython
import argparse
import dataloaders

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from os.path import join
from tqdm import tqdm
from torchvision import transforms

from util import Phi
from models import CSCTV, CSCDVTV
from models import CSCTVParams, CSCDVTVParams

# 同じ設定で学習したモデルが存在する場合に新たに学習を行う
MODEL_RESET = False
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
parser.add_argument("--train", type=bool, dest="train", help="number of channels", default=True)
parser.add_argument("--n_ch", type=int, dest="n_ch", help="number of channels", default=3)
parser.add_argument("--num_layer", type=int, dest="num_layer", help="Number of LISTA unfoldings", default=20)
parser.add_argument("--num_filters", type=int, dest="num_filters", help="Number of filters", default=64)
parser.add_argument("--kernel_size", type=int, dest="kernel_size", help="The size of the kernel", default=7)
parser.add_argument("--weight", type=float, dest="weight", help="Weight of DVTV", default=0.5)
parser.add_argument("--deg", type=str, dest="deg", help="The kind of the degradation.", default="debluring")
parser.add_argument("--blur_level", type=int, dest="blur_level", help="Should be an int in the range [0,255]", default=3)
parser.add_argument("--missing_rate", type=int, dest="missing_rate", help="Should be an int in the range [0,100]", default=50)
parser.add_argument("--noise_level", type=float, dest="noise_level", help="Should be an int in the range [0,255]", default=0)
parser.add_argument("--lr", type=float, dest="lr", help="ADAM Learning rate", default=1e-4)
parser.add_argument("--lr_step", type=int, dest="lr_step", help="Learning rate decrease step", default=50)
parser.add_argument("--lr_decay", type=float, dest="lr_decay", help="ADAM Learning rate decay (on step)", default=0.7)
parser.add_argument("--num_epochs", type=int, dest="num_epochs", help="Total number of epochs to train", default=250)
parser.add_argument("--batch_size", type=int, dest="batch_size", help="Total number of epochs to train", default=5)
parser.add_argument("--patch_size", type=int, dest="patch_size", help="Total number of epochs to train", default=64)
parser.add_argument("--out_dir", type=str, dest="out_dir", help="Results' dir path", default='trained_models')
parser.add_argument("--data_path", type=str, dest="data_path", help="Path to the dir containing the training and testing datasets.", default="./datasets/CBSD/")
parser.add_argument("--model_tag", type=str, dest="model_tag", help="Deteal of the experiment.", default=None)
args = parser.parse_args()

###################################################################################
# Setting Parameters
###################################################################################
train = args.train
n_channels = args.n_ch
K = args.num_layer
n_filters = args.num_filters
kernel_size = args.kernel_size
DVTV_weight = args.weight
deg = args.deg
blur_level = args.blur_level
missing_rate = args.missing_rate
noise_std = args.noise_level
lr = args.lr
lr_step = args.lr_step
lr_decay = args.lr_decay
epochs = args.num_epochs
batch_size = args.batch_size
patch_size = args.patch_size
model_tag = args.model_tag
data_path = args.data_path
to_pil = transforms.ToPILImage()

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

###################################################################################
# Initialize Result arrays and folders
###################################################################################
if deg == "denoising":
    setting = f"std{noise_std}_{model_tag}"
elif deg == "deblurring":
    setting = f"blur{blur_level}_std{noise_std}_{model_tag}"
elif deg == "inpainting":
    setting = f"rate{missing_rate}_std{noise_std}_{model_tag}"
else:
    raise ValueError(f"Degtype is not available for use. Got: '{deg}'")

os.makedirs(f"trained_models/{model_name}/{deg}", exist_ok=True)
os.makedirs(f"result/{model_name}/{deg}/{setting}", exist_ok=True)
best_loss = 1e8
loss_arr = []
psnr = np.zeros(epochs)
ssim = np.zeros(epochs)
    
###################################################################################
# Get Dataset
###################################################################################
train_path = [f'{data_path}/train']
valid_path = [f'{data_path}/valid']
test_path = [f'{data_path}/test']
loaders = dataloaders.get_dataloaders([train_path, valid_path, test_path], n_channels, patch_size, batch_size)

###################################################################################
# Create config file of setting
###################################################################################
config_dict = {
    'n_channel':n_channels,
    'kernel_size':kernel_size,
    'deg':deg,
    'w':DVTV_weight,
    'blur_level': blur_level,
    'missing_rate':missing_rate,
    'noise_std': noise_std,
    'n_filters': n_filters,
    'lr':lr,
    'lr_step': lr_step,
    'lr_decay': lr_decay,
    'layer': K,
    'epochs': epochs,
    'batch_size': batch_size,
    'patch_size': patch_size
    }

print("setting:",config_dict)
with open(f'trained_models/{model_name}/{deg}/{setting}.config','w') as txt_file:
    txt_file.write(str(config_dict))
    
###################################################################################
# Training
###################################################################################
print(f'(Training model...)')
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)
# Train and Validation
for epoch in tqdm(range(epochs)):
    for phase in ['train', 'valid']:

        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        
        for batch in loaders[phase]:
            batch = batch.cuda()
            phi = Phi(deg, batch, blur_level, missing_rate)
            deg_batch = phi(batch)
            noise = torch.randn_like(batch, device="cuda:0")*noise_std
            z = deg_batch + noise

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            with torch.set_grad_enabled(phase == 'train'):
                x_hat = model(z, phi, alpha=noise_std)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss = loss_func(x_hat, batch)
                    loss.backward()
                    optimizer.step()
                    # パラメータの非負制約
                    for name, param in model.state_dict().items():
                        if name in NONEGAPARAM:
                            param.data.clamp_(min=0.0)
                    loss_arr.append(loss.item())
                    if best_loss > loss.item():
                        torch.save(model.state_dict(), f'trained_models/{model_name}/{deg}/{setting}_best.model')
                        best_loss = loss.item()
                    torch.save(model.state_dict(), f'trained_models/{model_name}/{deg}/{setting}.model')
                    # util.plot_param(model, f'result/{model_name}/{deg}/{setting}')
                
                if phase == 'valid':
                    plt.clf()
                    img_arr = [to_pil(batch[0]), to_pil(util.ProxBoxConstraint(z[0])), to_pil(x_hat[0])]
                    img_name = ["Original", "Degradation", "Output"]
                    for i in range(3):
                        if n_channels==1:
                            plt.subplot(1,3,i+1).imshow(img_arr[i], cmap="gray")
                        else:
                            plt.subplot(1,3,i+1).imshow(img_arr[i])
                        plt.axis("off")
                        plt.grid(False)
                        plt.suptitle(f"Validation Result(Epoch{epoch+1})", y=0.7)
                        if i == 2:
                            psnr_ = cv2.PSNR(util.torch2numpy(batch), util.torch2numpy(x_hat), R=1.0)
                            ssim_ = util.my_ssim(util.torch2numpy(batch), util.torch2numpy(x_hat))
                            plt.text(-x_hat.shape[3]*0.1, x_hat.shape[2]*1.15, "PSNR:{:.2f}, SSIM:{:.4f}".format(psnr_, ssim_))
                    plt.savefig(f"result/{model_name}/{deg}/{setting}/valid_result.png")

                    # statistics
                    psnr[epoch] += cv2.PSNR(util.torch2numpy(batch), util.torch2numpy(x_hat), R=1.0)
                    ssim[epoch] += util.my_ssim(util.torch2numpy(batch), util.torch2numpy(x_hat))
        # print(model.lam1)
        psnr[epoch] /= len(loaders[phase])
        ssim[epoch] /= len(loaders[phase])
        if phase == "train":
            with open(f'result/{model_name}/{deg}/{setting}/loss.txt','w') as loss_file:
                loss_file.write(f'{loss_arr},') 
                util.plot_array(loss_arr, title="Loss", x_label=f"Iteration (minibatch:{batch_size})", y_label="Loss", result_path=f'result/{model_name}/{deg}/{setting}', file_name=f"Loss")
        else:
            with open(f'result/{model_name}/{deg}/{setting}/PSNR_{phase}.txt','a') as psnr_file:
                psnr_file.write(f'{psnr[epoch]},') 
            with open(f'result/{model_name}/{deg}/{setting}/SSIM_{phase}.txt','a') as ssim_file:
                ssim_file.write(f'{ssim[epoch]},')   
            util.plot_array(psnr[:epoch+1], title="PSNR", x_label="Epoch", y_label="PSNR", result_path=f'result/{model_name}/{deg}/{setting}', file_name=f"PSNR_{phase}")
            util.plot_array(ssim[:epoch+1], title="SSIM", x_label="Epoch", y_label="SSIM", result_path=f'result/{model_name}/{deg}/{setting}', file_name=f"SSIM_{phase}")
    scheduler.step()

###################################################################################
# Test
###################################################################################  
print(f"Testing model: channel {n_channels} with blur_level {blur_level} and noise_std {noise_std} on test images...")

model.eval()   
num_iters = 0
psnr = 0
ssim = 0

os.makedirs(f"result/{model_name}/{deg}/{setting}/org", exist_ok=True)
os.makedirs(f"result/{model_name}/{deg}/{setting}/deg", exist_ok=True)
os.makedirs(f"result/{model_name}/{deg}/{setting}/output", exist_ok=True)

for batch in tqdm(loaders['test']):
    phi = Phi(deg, batch, blur_level, missing_rate)
    
    batch = batch.cuda()
    deg_batch = phi(batch)
    noise = torch.randn_like(batch, device="cuda:0")*noise_std
    z = deg_batch + noise

    # forward
    with torch.set_grad_enabled(False):
        x_hat = model(z, phi, alpha=noise_std)

    org_img, deg_img, output = to_pil(batch[0]), to_pil(util.ProxBoxConstraint(z[0])), to_pil(x_hat[0])
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
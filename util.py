import math
import os
import cv2
import torch
import IPython
import openpyxl
import japanize_matplotlib

import matplotlib.pyplot as plt

from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim

'''
Prox計算
'''
def ProxL1(x, sigma):
    return torch.sign(x)*torch.max(torch.abs(x) - torch.ones_like(x)*sigma, torch.zeros_like(x))

def ProxL12(x, sigma):
    eps = 1e-8 #ゼロ除算を防ぐ
    th = torch.max(torch.tensor(1) - sigma/(torch.sqrt(torch.sum(x**2, dim=1, keepdim=True)+eps)), torch.tensor(0))
    return th*x

def ProxDVTVnorm(x, sigma, w):
    eps = 1e-8
    thL = torch.max(torch.tensor(1) - w*sigma/(torch.sqrt(torch.sum(x[:,:,:1]**2, dim=(1,2), keepdim=True)+eps)), torch.tensor(0))
    thC = torch.max(torch.tensor(1) - sigma/(torch.sqrt(torch.sum(x[:,:,1:]**2, dim=(1,2), keepdim=True)+eps)), torch.tensor(0))
    return torch.concat([thL*x[:,:,:1], thC*x[:,:,1:]], dim=2)

def ProxBoxConstraint(x):
    return torch.clamp(x,min=0, max=1.0)

def ProjL2ball(x, z, epsilon):
    diff_norm = torch.sqrt(torch.sum((x-z)**2, dim=(2,3), keepdim=True))
    return x*(diff_norm <= epsilon) + (z + (x-z)*(epsilon/diff_norm))*(diff_norm > epsilon)

'''
隣接差分計算
'''
def Dv(u):
    mb, channels, _, m = u.shape
    return torch.cat([torch.diff(u,dim=2), torch.zeros((mb,channels,1,m), device='cuda:0')], dim=2)

def Dvt(u):
    return torch.cat([-u[:,:,0,:].unsqueeze(2), -torch.diff(u, dim=2)], dim=2)

def Dh(u):
    mb, channels, n, _ = u.shape
    return torch.cat([torch.diff(u,dim=3), torch.zeros((mb, channels, n, 1), device='cuda:0')], dim=3)

def Dht(u):
    return torch.cat([-u[:,:,:,0].unsqueeze(3), -torch.diff(u, dim=3)], dim=3)

def D(u):
    return torch.stack((Dv(u), Dh(u)), dim=1)

def Dt(u):
    return Dvt(u[:,0])+Dht(u[:,1]) 

'''
畳み込み計算
'''
def conv(X, fil):
    '''
    X(img) : mb x channels x N x M 
    fil : k1 x k2
    '''
    _, _, N, M = X.shape
    k1, k2 = fil.shape
    center_k1 = math.ceil(k1/2)
    center_k2 = math.ceil(k2/2)
    fft_fil = torch.zeros(N, M, device="cuda:0")
    fft_fil[0:k1,0:k2] = torch.flip(fil, [0, 1])
    fft_fil = torch.roll(fft_fil, shifts=(N-center_k1, M-center_k2), dims=(0, 1))
    fft_fil = torch.fft.fft2(fft_fil)
    fft_X = torch.fft.fft2(X)
    B = torch.fft.ifft2(fft_fil*fft_X)
    return B.real

def convT(X, fil):
    '''
    X(img) : mb x channels x N x M 
    fil : k1 x k2
    '''
    _, _, N, M = X.shape
    k1, k2 = fil.shape
    center_k1 = math.ceil(k1/2)
    center_k2 = math.ceil(k2/2)
    fft_fil = torch.zeros(N, M, device="cuda:0")
    fft_fil[0:k1,0:k2] = torch.flip(fil, [0, 1])
    fft_fil = torch.roll(fft_fil, shifts=(N-center_k1, M-center_k2), dims=(0, 1))
    fft_fil = torch.fft.fft2(fft_fil)
    fft_X = torch.fft.ifft2(X)
    B = torch.fft.fft2(fft_fil*fft_X)
    return B.real

def conv_CSC(X, fil): #畳み込みの関数
    '''
    input(x) : mb x 特徴量 x channels x N  x  M
      filter : 特徴量 x channels x kernel_size x kernel_size
      return : mb x channels x  N  x  M
    '''
    _, _, channels, N, M = X.shape
    P, _, k1, k2 = fil.shape
    center_k1 = math.ceil(k1/2)
    center_k2 = math.ceil(k2/2)
    fft_fil = torch.zeros(P, channels, N, M, device='cuda:0')
    fft_fil[:, :, 0:k1, 0:k2] = torch.flip(fil, [2,3])
    fft_fil = torch.roll(fft_fil, shifts=(N-center_k1, M-center_k2), dims=(2, 3))
    fft_fil = torch.fft.fft2(fft_fil)
    fft_X = torch.fft.fft2(X)
    B = torch.fft.ifft2(fft_fil*fft_X)

    return torch.sum(B.real, dim=1)

def convT_CSC(X, fil): #転置畳み込みの関数
    '''
    input(x) : mb x channels x  M  x  N
      filter : 特徴量 x channels x kernel_size x kernel_size
      return : mb x channels x  特徴量 x  M  x  N
    '''
    _, _, N, M = X.shape
    P, channels, k1, k2 = fil.shape
    center_k1 = math.ceil(k1/2)
    center_k2 = math.ceil(k2/2)
    fft_fil = torch.zeros(P, channels, N, M, device='cuda:0')
    fft_fil[:, :, 0:k1, 0:k2] = torch.flip(fil, [2,3])
    fft_fil = torch.roll(fft_fil, shifts=(N-center_k1, M-center_k2), dims=(2, 3))
    
    fft_fil = torch.fft.fft2(fft_fil)
    fft_X = torch.fft.ifft2(X)
    B = torch.fft.fft2(fft_fil*fft_X.unsqueeze(1))

    return B.real

'''
保存系
'''
def plot_img(u, path_name):
    save_image(u, f"{path_name}.png")
    
def plot_fil(B):
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(8):
        for j in range(8):
            ax = axes[i,j]
            ax.imshow(B[i*8+j], cmap='gray')
            ax.axis('off')

def plot_result(x, z, x_hat, result_path, file_name, title=""):
    # x, z, x_hat : numpy(c x X x Y)
    z_psnr = cv2.PSNR(x, z, R=1.0)
    x_hat_psnr = cv2.PSNR(x, x_hat, R=1.0)
    z_ssim = ssim(x, z, data_range = 1.0, channel_axis=2)
    x_hat_ssim = ssim(x, x_hat, data_range = 1.0, channel_axis=2)
    plt.clf()
    #　原画像の出力
    plt.subplot(1,3,1).imshow(x)
    plt.title("原画像")
    plt.axis("off")
    plt.grid(False)
    #　劣化画像の出力
    plt.subplot(1,3,2).imshow(z)
    plt.title("劣化画像")
    plt.text(-5, x.shape[0]+40, "PSNR:{:.2f}, SSIM:{:.4f}".format(z_psnr, z_ssim))
    plt.grid(False)
    plt.axis("off")
    #　復元画像の出力
    plt.subplot(1,3,3).imshow(x_hat)
    plt.title("復元画像")
    # plt.xlabel("PSNR:{:.2f}, SSIM:{:.2f}".format(x_hat_psnr, x_hat_ssim))
    plt.text(-5, x.shape[0]+40, "PSNR:{:.2f}, SSIM:{:.4f}".format(x_hat_psnr, x_hat_ssim))
    plt.grid(False)
    plt.axis("off")

    plt.suptitle(title, y=0.9)
    path = os.path.join(result_path,f"{file_name}.png")
    plt.savefig(path, bbox_inches='tight')

def plot_array(array, title, x_label, y_label, result_path, file_name): #loss遷移の描画
    plt.clf()
    plt.plot(torch.arange(1, len(array)+1, 1).cpu(), array)
    plt.xlabel(f"{x_label}")
    plt.ylabel(f"{y_label}")
    plt.title(f"{title}")
    path = os.path.join(result_path, f"{file_name}.png")
    plt.savefig(path, bbox_inches='tight')

def plot_dist_to_epsilon(fidelity, epsilon, result_path):
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(torch.arange(0, len(fidelity), 1).cpu(), fidelity, color='b', label=r"$\|Ax-z\|_2$")
    ax.plot(torch.arange(0, len(epsilon), 1).cpu(), epsilon, color='r', label=r"$\epsilon$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|Ax-z\|_2$", color='b')
    path = os.path.join(result_path, "dist_to_epsilon.png")
    plt.legend()
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def init_workbook(parameters):
    workbook = openpyxl.Workbook()
    for param_name, param in parameters.items():
        new_sheet = workbook.create_sheet(title=param_name)
        new_sheet.append(param.tolist())
    return workbook

def save_params(workbook, parameters, model_path):
    for param_name, param in parameters.items():
        sheet = workbook.get_sheet_by_name(param_name)
        sheet.append(param.tolist())
    workbook.save(os.path.join(model_path,'parameters.xlsx'))
    return workbook
'''
その他
'''
def torch2numpy(x):
    # x : mb x channel x m x n
    return torch.permute(x,(0,2,3,1)).cpu().detach().numpy()

def my_ssim(batch_images1, batch_images2):
    ssim_values = []
    for img1, img2 in zip(batch_images1, batch_images2):
        # 各画像ペアに対してSSIMを計算する
        ssim_value = ssim(img1, img2, data_range = 1.0, channel_axis=2)  # multichannel=Trueは、カラー画像の場合に使用する
        ssim_values.append(ssim_value)
    return sum(ssim_values)/len(ssim_values)

def make_degmat(deg_type, blur_size=3):

    if deg_type == "denoising":
        kernel = torch.tensor([[1,0],[0,0]])
    # elif deg_type == "debluring":
    else:
        kernel = torch.ones(blur_size, blur_size)/(blur_size*blur_size)
        
    return kernel.cuda()

def calc_max_singular(fil): #畳み込みの関数
    '''
    input(x) : M  x  N
    filter : kernel_size x kernel_size
    '''
    M = N = 30
    k1, k2 = fil.shape
    center_k1 = math.ceil(k1/2)
    center_k2 = math.ceil(k2/2)
    fft_fil = torch.zeros(M, N)
    fft_fil[0:k1, 0:k2] = fil
    fft_fil = torch.roll(fft_fil, shifts=(N-center_k1, M-center_k2), dims=(0, 1))
    fft_fil = torch.fft.fft2(fft_fil)
    return torch.max(abs(fft_fil))

def calc_max_singular_P(fil): #畳み込みの関数
    '''
    input(x) : P  x  M  x  N
    filter : kernel_size x kernel_size
    '''
    P = fil.shape[0]
    result = 0
    for p in range(P):
        result += calc_max_singular(fil[p])**2
    return torch.sqrt(result)

def C(u):
    u1 = 1/math.sqrt(3)*(u[:,0]+u[:,1]+u[:,2]).unsqueeze(1)
    u2 = 1/math.sqrt(2)*(u[:,0]-u[:,2]).unsqueeze(1)
    u3 = 1/math.sqrt(6)*(u[:,0]-2*u[:,1]+u[:,2]).unsqueeze(1)
    return torch.cat([u1, u2, u3], dim=1) 

def Ct(u):
    u1 = ((1/math.sqrt(3))*u[:,0]+(1/math.sqrt(2))*u[:,1]+(1/math.sqrt(6))*u[:,2]).unsqueeze(1)
    u2 = ((1/math.sqrt(3))*u[:,0]-(2/math.sqrt(6))*u[:,2]).unsqueeze(1)
    u3 = ((1/math.sqrt(3))*u[:,0]-(1/math.sqrt(2))*u[:,1]+(1/math.sqrt(6))*u[:,2]).unsqueeze(1)
    return torch.cat([u1, u2, u3], dim=1) 
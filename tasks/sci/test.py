# -*- coding: utf-8 -*-
import numpy as np
from tfpnp.pnp.denoiser import SCIUNetDenoiser
from until import *
import torch
from scipy.io import loadmat
import os
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from tfpnp.utils.misc import torch2img255
from skimage.restoration import denoise_tv_chambolle


def show(output, gt):
    output = output.cpu().numpy()
    gt_ = gt.clone().cpu().numpy()
    output = (np.clip(output, 0, 1)*255).astype(np.uint8)
    gt_ = (np.clip(gt_, 0, 1)*255).astype(np.uint8)
    for i in range(0, 24, 6):
        img = Image.fromarray(output[i, :, :])
        plt.subplot(4, 2, 1 + 2*(i//6))
        plt.imshow(img)
        img = Image.fromarray(gt_[i, :, :])
        plt.subplot(4, 2, 2 + 2*(i//6))
        plt.imshow(img)
    plt.show()


def A_CHW(x, Phi):
    '''
    Forward model of snapshot compressive imaging (SCI), where multiple coded
    frames are collapsed into a snapshot measurement.
    '''
    return torch.sum(x * Phi, dim=0)  # element-wise product


def psnr(ref, img):
    '''
    Peak signal-to-noise ratio (PSNR).
    '''
    mse = torch.mean( (ref - img) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.
    #PIXEL_MAX = ref.max()
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

denoiser = SCIUNetDenoiser().to('cuda')

def test_date():
    base_dir = os.path.join("data", "demo_data")
    datasetdir = os.path.join(base_dir, "Dataset")
    resultsdir = os.path.join(base_dir, "results")
    maskdir = os.path.join(base_dir, "mask")

    datname = 'scene01'  # name of the dataset
    matfile = os.path.join("data", "CAVE_512_28", f"{datname}.mat")  # path of the .mat data file

    ## data operation
    r, c, nC, step = 256, 256, 28, 2
    mask = np.zeros((r, c + step * (nC - 1)))
    mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, nC))
    mask256 = loadmat(os.path.join(maskdir, "mask256.mat"))['mask']
    for i in range(nC):
        mask_3d[:, i:i + 256, i] = mask256
    truth = loadmat(matfile)['data_slice'][100:356, 100:356, :]
    truth = (truth - truth.min()) / (truth.max() - truth.min())

    truth_shift = np.zeros((r, c + step * (nC - 1), nC))
    for i in range(nC):
        truth_shift[:, i * step:i * step + 256, i] = truth[:, :, i]
    meas = np.sum(mask_3d * truth_shift, 2)
    plt.figure()
    plt.imshow(meas, cmap='gray')
    plt.savefig(os.path.join(resultsdir, "result_img", f"{datname}_meas.png"))
    Phi = mask_3d
    Phi_sum = np.sum(mask_3d ** 2, 2)
    Phi_sum[Phi_sum == 0] = 1

    inputs = [At(meas, Phi), Phi_sum[:, :, np.newaxis], Phi, truth_shift, meas[:, :, np.newaxis]]
    for i, input in enumerate(inputs):
        inputs[i] = torch.from_numpy(input).to('cuda').permute(2, 0, 1).squeeze()
    return inputs



mat = loadmat(os.path.join("data/val/0.mat"))
mat['output'] = mat['ATy0']
mat['input'] = mat['x0']
mat['name'] = 1
gt = torch.from_numpy(mat['gt']).to('cuda')

mask = torch.from_numpy(mat['mask']).to('cuda')
# variables, (y0, mask) = inputs
# sigma_d, mu = parameters
Phi_sum = torch.sum(mask ** 2, 0)
Phi_sum[Phi_sum == 0] = 1

y0 = torch.from_numpy(mat['y0']).to('cuda')
theta = torch.from_numpy(mat['ATy0']).to('cuda')



# (theta, Phi_sum, Phi, gt, y0) = test_date()
# mask = Phi



b = torch.zeros_like(theta).to('cuda')
gamma = 0.01
_lambda = 1
s_gt = shift_back_CHW(gt.clone(), 2)
s_y0 = shift_back_CHW(theta.clone(), 2)
show(s_gt.clone(), s_y0.clone())
print(psnr(s_gt, s_y0))
for i in range(80):
    yb = A_CHW(theta + b, mask)
    x = (theta + b) + _lambda * (At_CHW((y0.squeeze() - yb) / (Phi_sum + gamma), mask))  # ADMM
    x1 = shift_back_CHW(x - b, 2)

    # theta = denoiser(theta.unsqueeze(0), torch.tensor([0.1]).to('cuda')).squeeze(0)
    theta = denoise_tv_chambolle(np.array(x1.permute(1, 2, 0).cpu()), 0.1, n_iter_max=5, multichannel=True)
    theta = torch.from_numpy(theta).to('cuda').permute(2, 0, 1)
    theta = shift_CHW(theta, 2)
    b = b - (x - theta)  # update residual


    s_theta = shift_back_CHW(theta.clone(), 2)
    # show(s_theta.clone(), s_gt.clone())
    p = psnr(s_gt, s_theta)
    print(p)

show(s_theta.clone(), s_gt.clone())
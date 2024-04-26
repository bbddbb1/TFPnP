import os
import torch
from .models.unet import UNet

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class UNetDenoiser2D(torch.nn.Module):
    def __init__(self, ckpt_path=None):
        super().__init__()
        if ckpt_path is None:
            ckpt_path = os.path.join(CURRENT_DIR, 'pretrained', 'unet-nm.pt')
            if not os.path.exists(ckpt_path):
                raise ValueError('Default ckpt not found, you have to provide a ckpt path')
            
        net = UNet(2, 1)
        net.load_state_dict(torch.load(ckpt_path))
        net.eval()
        for param in net.parameters():
            param.requires_grad = False

        self.net = net

    def forward(self, x, sigma):
        # x: [B,1,H,W]
        N, C, H, W = x.shape

        sigma = sigma.view(N, 1, 1, 1)

        noise_map = torch.ones(N, 1, H, W).to(x.device) * sigma
        out = self.net(torch.cat([x, noise_map], dim=1))

        return torch.clamp(out, 0, 1)


class SCIUNetDenoiser(torch.nn.Module):
    def __init__(self, ckpt_path=None):
        super().__init__()
        if ckpt_path is None:
            ckpt_path = os.path.join(CURRENT_DIR, 'pretrained', 'unet-nm.pt')
            if not os.path.exists(ckpt_path):
                raise ValueError('Default ckpt not found, you have to provide a ckpt path')

        net = UNet(2, 1)
        net.load_state_dict(torch.load(ckpt_path))
        net.eval()
        for param in net.parameters():
            param.requires_grad = False

        self.net = net

    def forward(self, x, sigma):
        # x: [B,C,H,W]
        N, C, H, W = x.shape

        sigma = sigma.view(N, 1, 1, 1)
        noise_map = torch.ones(N, 1, H, W).to(x.device) * sigma
        output = []
        for i in range(C):
            output.append(self.net(torch.cat([x[:, i:i+1, :, :], noise_map], dim=1)).clamp(0, 1).squeeze(1))
        return torch.stack(output, dim=1)


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 14:30:46 2019
Last edited on 06/ Nov/ 2019

author: Wei-Chung

description: this is the denoise function "denoise_tv_chambolle" in skimage.
It only supports numpy array, this function transfer it and it support torch.tensor.
"""

import torch


# %%
def diff(image, axis):
    '''
    Take the difference of different dimension(1~4) of images
    '''
    ndim = image.ndim
    if ndim == 3:
        if axis == 0:
            return image[1:, :, :] - image[:-1, :, :]
        elif axis == 1:
            return image[:, 1:, :] - image[:, :-1, :]
        elif axis == 2:
            return image[:, :, 1:] - image[:, :, :-1]

    elif ndim == 2:
        if axis == 0:
            return image[1:, :] - image[:-1, :]
        elif axis == 1:
            return image[:, 1:] - image[:, :-1]
    elif ndim == 4:
        if axis == 0:
            return image[1:, :, :, :] - image[:-1, :, :, :]
        elif axis == 1:
            return image[:, 1:, :, :] - image[:, :-1, :, :]
        elif axis == 2:
            return image[:, :, 1:, :] - image[:, :, :-1, :]
        elif axis == 3:
            return image[:, :, :, 1:] - image[:, :, :, :-1]
    elif ndim == 1:
        if axis == 0:
            return image[1:] - image[:-1]


def _denoise_tv_chambolle_nd_torch(image, weight=0.1, eps=2.e-4, n_iter_max=200):
    """
    image : torch.tensor
        n-D input data to be denoised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.
    Returns
    -------
    out : torch.tensor
        Denoised array of floats.

    """

    ndim = image.ndim
    pt = torch.zeros((image.ndim,) + image.shape, dtype=image.dtype).to('cuda')
    gt = torch.zeros_like(pt).to('cuda')
    dt = torch.zeros_like(image).to('cuda')
    i = 0
    while i < n_iter_max:
        if i > 0:
            # dt will be the (negative) divergence of p
            dt = -pt.sum(0)
            slices_dt = [slice(None), ] * ndim
            slices_pt = [slice(None), ] * (ndim + 1)
            for ax in range(ndim):
                slices_dt[ax] = slice(1, None)
                slices_pt[ax + 1] = slice(0, -1)
                slices_pt[0] = ax
                dt[tuple(slices_dt)] += pt[tuple(slices_pt)]
                slices_dt[ax] = slice(None)
                slices_pt[ax + 1] = slice(None)
            out = image + dt
        else:
            out = image
        Et = torch.mul(dt, dt).sum()

        # gt stores the gradients of out along each axis
        # e.g. gt[0] is the first order finite difference along axis 0
        slices_gt = [slice(None), ] * (ndim + 1)
        for ax in range(ndim):
            slices_gt[ax + 1] = slice(0, -1)
            slices_gt[0] = ax
            gt[tuple(slices_gt)] = diff(out, ax)
            slices_gt[ax + 1] = slice(None)

        norm = torch.sqrt((gt ** 2).sum(axis=0)).unsqueeze(0)
        Et = Et + weight * norm.sum()
        tau = torch.tensor(1. / (2. * ndim)).to('cuda')
        norm = norm * tau / weight
        norm = norm + 1.
        pt = pt - tau * gt
        pt = pt / norm
        Et = Et / float(image.view(-1).shape[0])
        if i == 0:
            E_init = Et
            E_previous = Et
        else:
            if torch.abs(E_previous - Et) < eps * E_init:
                break
            else:
                E_previous = Et
        i += 1

    return out


def denoise_tv_chambolle_torch(image, weight=0.1, eps=2.e-4, n_iter_max=200,
                               multichannel=True):
    """Perform total-variation denoising on n-dimensional images.
    Parameters
    ----------
    image : torch.tensor of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an torch.tensor of floats for the computation
        of the denoised image.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that
        determines the stop criterion. The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.
    multichannel : bool, optional
        Apply total-variation denoising separately for each channel. This
        option should be true for color images, otherwise the denoising is
        also applied in the channels dimension.
    Returns
    -------
    out : torch.tensor
        Denoised image.

    """
    im_type = (image.cpu().numpy()).dtype
    if not im_type.kind == 'f':
        image = image.type(torch.float64)
        image = image / torch.abs(image.max() + image.min())

    if multichannel:
        out = torch.zeros_like(image)
        for c in range(image.shape[-1]):
            out[..., c] = _denoise_tv_chambolle_nd_torch(image[..., c], weight, eps, n_iter_max)
    else:
        out = _denoise_tv_chambolle_nd_torch(image, weight, eps, n_iter_max)

    return out

class TvChambolleDenoiser(torch.nn.Module):
    def forward(self, image, weight, eps=2.e-4, n_iter_max=10):
        image = image.clone().detach()
        weight = weight.clone().detach()
        image = image.permute(0, 2, 3, 1)
        for i in range(image.shape[0]):
            image[i, ...] = denoise_tv_chambolle_torch(image[i, ...], weight[i], eps, n_iter_max)
        return image.permute(0, 3, 1, 2)
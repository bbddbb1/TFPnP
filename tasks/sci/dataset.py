import os
import numpy as np
from PIL import Image
from until import A, At
import torch 
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat

from tfpnp.data.util import scale_height, scale_width, data_augment
from tfpnp.utils import transforms
from tfpnp.utils.transforms import complex2real
from until import A, At, shift, shift_back, At_CHW, shift_back_CHW
import random


def prepare_data_cave(path, file_num):
    HR_HSI = np.zeros((((512,512,28,file_num))))
    file_list = os.listdir(path)
    # for idx in range(1):
    for idx in range(file_num):
        print(f'loading CAVE {idx}')
        ####  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path, HR_code)
        data = loadmat(path1)
        HR_HSI[:,:,:,idx] = data['data_slice']
        # HR_HSI[HR_HSI < 0] = 0
        # HR_HSI[HR_HSI > 1] = 1
    return HR_HSI

class SCIDataset(Dataset):
    def __init__(self, mask_dir, CAVE, noise_model=None, size=None, target_size=None, repeat=1, augment=False, train=True):
        super().__init__()
        self.noise_model = noise_model
        self.size = size
        self.repeat = repeat
        self.target_size = target_size
        self.augment = augment
        data = loadmat(os.path.join(mask_dir, "mask.mat"))
        self.mask = data['mask']
        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 28))
        self.CAVE = CAVE
        self.isTrain = train

    def __getitem__(self, index):
        hsi = self.CAVE[:, :, :, index]
        shape = np.shape(hsi)

        px = random.randint(0, shape[0] - self.size)
        py = random.randint(0, shape[1] - self.size)
        label = hsi[px:px + self.size:1, py:py + self.size:1, :]
        label = (label - label.min()) / (label.max() - label.min())

        pxm = random.randint(0, 660 - self.size)
        pym = random.randint(0, 660 - self.size)
        mask_3d = self.mask_3d[pxm:pxm + self.size:1, pym:pym + self.size:1, :]

        mask_3d_shift = shift(mask_3d, 2)
        mask_3d_shift_s = np.sum(mask_3d_shift ** 2, axis=2, keepdims=False)
        mask_3d_shift_s[mask_3d_shift_s == 0] = 1

        if self.isTrain:
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                label = np.rot90(label)

            # Random vertical Flip
            for j in range(vFlip):
                label = label[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                label = label[::-1, :, :].copy()

        label = shift(label, 2)
        temp = mask_3d_shift * label
        meas = np.sum(temp, axis=2)

        label = torch.FloatTensor(label).permute(2, 0, 1)
        mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
        mask_3d_shift_s = torch.FloatTensor(mask_3d_shift_s.copy())
        y0 = torch.FloatTensor(meas)
        sigma_n = 0
        target = label
        # if self.noise_model is not None:
        #     y0, sigma_n = self.noise_model(y0)
        ATy0 = At_CHW(y0, mask_3d_shift)
        x0 = ATy0.clone().detach()
        mask_sum = mask_3d_shift_s
        output = ATy0.clone().detach()
        mask = mask_3d_shift
        y0.unsqueeze_(0)
        sigma_n = np.ones_like(y0) * sigma_n
        
        dic = {'y0': y0, 'x0': x0, 'ATy0': ATy0, 'gt': target, 'mask': mask, 'sigma_n': sigma_n, 'output': output, 'input': x0}

        # y0,x0,ATy0, sigma_n: C, W, H
        # gt, output: C, W, H
        # mask: 1, W, H
        
        return dic

    def __len__(self):
        return self.CAVE.shape[3]


class SCIEvalDataset(Dataset):
    def __init__(self, dirs):
        super(SCIEvalDataset, self).__init__()
        self.dirs = dirs

    def __getitem__(self, index):
        mat = loadmat(os.path.join(f"{self.dirs}", f"{index}.mat"))
        #
        # mat['name'] = mat['name'].item()
        # mat.pop('__globals__', None)
        # mat.pop('__header__', None)
        # mat.pop('__version__', None)
        mat['output'] = mat['ATy0']
        mat['input'] = mat['x0']
        mat['name'] = index
        return mat

    def __len__(self):
        return len(os.listdir(self.dirs))

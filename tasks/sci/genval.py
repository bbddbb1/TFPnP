import os

import numpy as np

from dataset import SCIDataset
#!/usr/bin/env python3
from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader
from scipy.io import loadmat
import scipy
from env import SCIEnv
from dataset import SCIDataset, SCIEvalDataset
from solver import create_solver_sci

from tfpnp.policy.sync_batchnorm import DataParallelWithCallback
from tfpnp.policy import create_policy_network
from tfpnp.pnp import create_denoiser
from tfpnp.trainer import MDDPGTrainer
from tfpnp.eval import Evaluator
from tfpnp.utils.noise import GaussianModelD
from tfpnp.utils.options import Options
from dataset import prepare_data_cave



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sampling_masks = ['radial_128_2', 'radial_128_4', 'radial_128_8']

data_dir = Path('data')
train_root = data_dir / 'CAVE_512_28'
cave = prepare_data_cave(train_root, 4)
sigma_ns = [5, 10, 15]
noise_model = GaussianModelD(sigma_ns)
mask_dir = data_dir / 'mask'
train_dataset = SCIDataset(mask_dir, cave, noise_model=noise_model, size=256, train=False)


for i in range(4):
    d = train_dataset.__getitem__(i)
    for k, v in d.items():
        d[k] = np.array(d[k])

    scipy.io.savemat(f'data/val/{i}.mat', d)

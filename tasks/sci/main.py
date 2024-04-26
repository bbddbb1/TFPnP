#!/usr/bin/env python3
from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader
from scipy.io import loadmat

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

def build_evaluator(data_dir, solver, sigma_n, save_dir):
    # val_loaders = {}
    # for sampling_mask in sampling_masks:
    #     root = data_dir / 'CAVE_512_28'
    #     dataset = SCIEvalDataset(root)
    #     loader = DataLoader(dataset, batch_size=1, shuffle=False)
    #     val_loaders[f'{sampling_mask}_{sigma_n}'] = loader

    val_loaders = {}
    dataset = SCIEvalDataset("data/val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    val_loaders[f'test'] = loader
    eval_env = SCIEnv(None, solver, max_episode_step=opt.max_episode_step).to(device)
    evaluator = Evaluator(eval_env, val_loaders, save_dir)
    return evaluator


def train(opt, data_dir, mask_dir, policy, solver, log_dir):
    sigma_ns = [5, 10, 15]
    noise_model = GaussianModelD(sigma_ns)
    cave = prepare_data_cave(train_root, 16)
    train_dataset = SCIDataset(mask_dir, cave, noise_model=noise_model, size=256)
    train_loader = DataLoader(train_dataset, opt.env_batch,
                              shuffle=True, num_workers=opt.num_workers,
                              pin_memory=True, drop_last=True)

    eval = build_evaluator(data_dir, solver, '15', log_dir / 'eval_results')
    
    env = SCIEnv(train_loader, solver, max_episode_step=opt.max_episode_step).to(device)

    def lr_scheduler(step):
        if step < 10000:
            return {'critic': 3e-4, 'actor': 1e-3}
        else:
            return {'critic': 1e-4, 'actor': 3e-4}

    trainer = MDDPGTrainer(opt, env, policy,
                           lr_scheduler=lr_scheduler, 
                           device=device,
                           log_dir=log_dir,
                           evaluator=eval, 
                           enable_tensorboard=True)
    if opt.resume:
        trainer.load_model(opt.resume, opt.resume_step)
    trainer.train()


def main(opt):
    data_dir = Path('data')
    log_dir = Path(opt.output)
    mask_dir = data_dir / 'mask'

    base_dim = SCIEnv.ob_base_dim
    policy = create_policy_network(opt, 112).to(device)  # policy network
    denoiser = create_denoiser(opt).to(device)
    solver = create_solver_sci(opt, denoiser).to(device)
    if torch.cuda.device_count() > 1:
        solver = DataParallelWithCallback(solver)
        
    if opt.eval:
        ckpt = torch.load(opt.resume)
        policy.load_state_dict(ckpt)
        for sigma_n in [5, 10, 15]:
            save_dir = log_dir / 'test_results' / str(sigma_n)
            e = build_evaluator(data_dir, solver, sigma_n, save_dir)
            e.eval(policy, step=opt.resume_step)
            print('--------------------------')
        return 
    
    train(opt, data_dir, mask_dir, policy, solver, log_dir)

if __name__ == "__main__":
    option = Options()
    opt = option.parse()
    main(opt)

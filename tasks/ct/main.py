#!/usr/bin/env python3
import torch
import torch.utils.data.dataloader
from pathlib import Path

from env import CTEnv
from dataset import CT_transform, CTDataset
from solver import create_solver_ct

from tfpnp.data.dataset import ImageFolder
from tfpnp.policy.sync_batchnorm import DataParallelWithCallback
from tfpnp.policy import create_policy_network
from tfpnp.pnp import create_denoiser
from tfpnp.trainer import MDDPGTrainer
from tfpnp.eval import Evaluator
from tfpnp.utils.noise import GaussianModelP
from tfpnp.utils.options import Options


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(opt):
    data_dir = Path('data')
    log_dir = Path('log') / opt.exp

    view = 30
    sigma_ns = [0.05, 0.075, 0.10]

    base_dim = CTEnv.ob_base_dim
    policy = create_policy_network(opt, base_dim).to(device)  # policy network
    denoiser = create_denoiser(opt).to(device)
    solver = create_solver_ct(opt, denoiser).to(device)

    # ---------------------------------------------------------------------------- #
    #                                     Valid                                    #
    # ---------------------------------------------------------------------------- #

    val_root = data_dir / 'ct' / 'CT_test'
    val_datasets = [CTDataset(val_root, fns=None, view=view, noise_model=GaussianModelP([sigma_n])) for sigma_n in sigma_ns]

    val_loaders = [torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                               num_workers=0, pin_memory=False) for val_dataset in val_datasets]
    val_names = [f'CT_{sigma_n*100}' for sigma_n in sigma_ns]
    val_loaders = dict(zip(val_names, val_loaders))

    if torch.cuda.device_count() > 1:
        solver = DataParallelWithCallback(solver)

    eval_env = CTEnv(None, solver, max_episode_step=opt.max_episode_step).to(device)

    if opt.eval:
        evaluator = Evaluator(eval_env, val_loaders, savedir=log_dir / 'test_results')
        actor_ckpt = torch.load(opt.resume)
        policy.load_state_dict(actor_ckpt)
        evaluator.eval(policy, step=opt.resume_step)
        return

    # ---------------------------------------------------------------------------- #
    #                                     Train                                    #
    # ---------------------------------------------------------------------------- #
    noise_model = GaussianModelP(sigma_ns)

    train_root = data_dir / 'Images_128'

    train_dataset = ImageFolder(train_root, fns=None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.env_batch, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    data_transform = CT_transform(view, noise_model)
    env = CTEnv(train_loader, solver, max_episode_step=opt.max_episode_step, data_transform=data_transform).to(device)

    def lr_scheduler(step):
        if step < 10000:
            return {'critic': 1e-4, 'actor': 5e-5}
        else:
            return {'critic': 5e-5, 'actor': 1e-5}

    evaluator = Evaluator(eval_env, val_loaders, savedir=log_dir / 'eval_results')
    trainer = MDDPGTrainer(opt, env, policy,
                           lr_scheduler=lr_scheduler,
                           device=device,
                           log_dir=log_dir,
                           evaluator=evaluator,
                           enable_tensorboard=True)
    trainer.train()


if __name__ == "__main__":
    option = Options()
    opt = option.parse()
    main(opt)

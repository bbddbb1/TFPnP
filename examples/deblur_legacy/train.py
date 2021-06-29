import torch
import torch.utils.data

from tensorboardX import SummaryWriter

from options import TrainOptions
from env import DeblurEnv
from dataset import HSIDeblurDataset
from solver import ADMMSolver_Deblur

from tfpnp.eval import Evaluator
from tfpnp.pnp.denoiser import GRUNetDenoiser
from tfpnp.policy.resnet import ResNetActor_HSI
from tfpnp.trainer.a2cddpg.critic import ResNet_wobn
from tfpnp.trainer import A2CDDPGTrainer

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer(A2CDDPGTrainer):
    def save_experience(self, ob):
        ob_detached = ob.clone().detach().cpu() # crucial to prevent out of gpu memory
        for i in range(ob_detached.shape[0]):
            self.buffer.store(ob_detached[i])
            
    def convert2batch(self, states):
        return torch.stack(states, dim=0).to(self.device)

    
def get_dataloaders(opt):
    train_dataset = HSIDeblurDataset('/media/exthdd/datasets/hsi/ECCVData/icvl_512_0', target_size=(128,128))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.env_batch)
    
    val_datasets = [HSIDeblurDataset('/media/exthdd/datasets/hsi/ECCVData/icvl_512_0', training=False, target_size=(128,128))]
    val_loaders = [torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True) for val_dataset in val_datasets]
    
    val_names = ['icvl']
    
    return train_loader, dict(zip(val_names, val_loaders))

def lr_scheduler(step):
    if step < 10000:
        lr = (3e-4, 1e-3)
    else:
        lr = (1e-4, 3e-4)
    return lr

if __name__ == "__main__":
    option = TrainOptions()
    opt = option.parse()
    
    writer = SummaryWriter('./log/{}/tensorboard'.format(opt.exp))

    train_loader, val_loaders = get_dataloaders(opt)

    policy_network = ResNetActor_HSI(189, opt.action_pack).to(device)
    critic = ResNet_wobn(189, 18, 1).to(device)
    critic_target = ResNet_wobn(189, 18, 1) .to(device)
    
    denoiser = GRUNetDenoiser().to(device)
    solver = ADMMSolver_Deblur(denoiser)
    env = DeblurEnv(train_loader, solver, max_step=opt.max_step, device=device)
    
    eval_env = DeblurEnv(None, solver, max_step=opt.max_step, device=device)
    evaluator = Evaluator(opt, eval_env, val_loaders, writer, device=device, savedir='./log/{}/eval'.format(opt.exp))
    
    trainer = Trainer(opt, env, policy_network=policy_network, 
                             critic=critic, critic_target=critic_target, 
                             device=device, evaluator=evaluator, writer=writer)
    
    trainer.train()
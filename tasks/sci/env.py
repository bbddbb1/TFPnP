import torch
from tfpnp.data.batch import Batch
from tfpnp.env import PnPEnv
from tfpnp.utils.misc import torch2img255
from tfpnp.utils.transforms import complex2channel, complex2real
from until import shift_back_CHW

class SCIEnv(PnPEnv):
    # class attribute: the dimension of ob (exclude solver variable)
    ob_base_dim = 6  
    def __init__(self, data_loader, solver, max_episode_step):
        super().__init__(data_loader, solver, max_episode_step)
    
    def get_policy_ob(self, ob):
        ob= torch.cat([
            ob.variables,
            ob.y0,
            ob.ATy0,
            ob.mask,
            ob.T,
            ob.sigma_n,
        ], 1)
        return ob
    
    def get_eval_ob(self, ob):
        return self.get_policy_ob(ob)
    
    def _get_attribute(self, ob, key):
        if key == 'gt':
            return ob.gt
        elif key == 'output':
            return self.solver.get_output(ob.variables)
        elif key == 'input':
            return ob.ATy0
        elif key == 'solver_input':
            return ob.variables, (ob.y0, ob.mask.bool())
        else:
            raise NotImplementedError('key is not supported, ' + str(key))
        
    def _build_next_ob(self, ob, solver_state):
        return Batch(gt=ob.gt,
                     y0=ob.y0,
                     ATy0=ob.ATy0,
                     variables=solver_state,
                     mask=ob.mask,
                     sigma_n=ob.sigma_n,
                     T=ob.T + 1/self.max_episode_step)
    
    def _observation(self):
        idx_left = self.idx_left
        return Batch(gt=self.state['gt'][idx_left, ...],
                     y0=self.state['y0'][idx_left, ...],
                     ATy0=self.state['ATy0'][idx_left, ...],
                     variables=self.state['solver'][idx_left, ...],
                     mask=self.state['mask'][idx_left, ...].float(),
                     sigma_n=self.state['sigma_n'][idx_left, ...],
                     T=self.state['T'][idx_left, ...])

    def get_images(self, ob, pre_process=torch2img255):
        input = self._get_attribute(ob, 'input')
        output = self._get_attribute(ob, 'output')
        gt = self._get_attribute(ob, 'gt')
        imgs = [input, output, gt]
        for i, img in enumerate(imgs):
            imgs[i] = shift_back_CHW(img.squeeze(), 2)
        input = pre_process(imgs[0])
        output = pre_process(imgs[1])
        gt = pre_process(imgs[2])
        return input, output, gt
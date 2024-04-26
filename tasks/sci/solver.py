import torch
import numpy as np

from tfpnp.pnp.solver.base import ADMMSolver, HQSSolver, PGSolver, APGSolver, REDADMMSolver, AMPSolver
from tfpnp.utils import transforms
from until import A_CHW, At_CHW, shift, shift_back, At_BCHW, shift_back_bcwh, shift_bcwh

# decorator
def complex2real_decorator(func):
    def real_func(*args, **kwargs):
        return transforms.complex2real(func(*args, **kwargs))
    return real_func


class SCIMixin:
    @complex2real_decorator
    def get_output(self, state):
        return super().get_output(state)

    def filter_aux_inputs(self, state):
        return (state['y0'], state['mask'])

class ADMMSolver_SCI(SCIMixin, ADMMSolver):
    #TODO warning: CSMRIMixin must be put behind the ADMMSolver class
    def __init__(self, denoiser):
        super().__init__(denoiser)

    def get_output(self, state):
        # x's shape [B,1,C,W,H]
        x, _ = torch.split(state, state.shape[1] // 2, dim=1)
        return x

    def reset(self, data):
        theta = data['x0'].clone().detach()
        b = torch.zeros_like(theta)
        return torch.cat((theta, b), dim=1)

    def filter_hyperparameter(self, action):
        return action['sigma_d'], action['gamma'], action['_lambda']

    def forward(self, inputs, parameters, iter_num=None):
        # y0:    [B,1,C,W,H]
        # mask:  [B,1,C,W,H]
        # x,z,u: [B,1,C,W,H]

        variables, (y0, mask) = inputs
        sigma_d, gamma, _lambda = parameters
        Phi_sum = mask.sum(1)
        Phi_sum[Phi_sum == 0] = 1
        theta, b = torch.split(variables, variables.shape[1] // 2, dim=1)

        # x0 = At(y, Phi)  # default start point (initialized value)
        # theta = x0
        # b = np.zeros_like(x0)

        # gamma = 0.01
        # _lambda = 1
        if iter_num is None:
            iter_num = sigma_d.shape[-1]
        for i in range(iter_num):
            yb = A_CHW(theta + b, mask)
            x = (theta + b) + _lambda[:, i, None, None, None] * (At_BCHW((y0.squeeze() - yb) / (Phi_sum + gamma[:, i, None, None]), mask))  # ADMM
            x1 = shift_back_bcwh(x - b, 2)

            theta = self.prox_mapping(x1, sigma_d[:, i].to('cuda'))
            theta = shift_bcwh(theta, 2)
            b = b - (x-theta)  # update residual
        torch.cuda.empty_cache()
        return torch.cat((theta, b), dim=1)


_solver_map = {
    'admm_sci': ADMMSolver_SCI,
}


def create_solver_sci(opt, denoiser):
    print(f'[i] use solver: {opt.solver}')
    
    if opt.solver in _solver_map:
        solver = _solver_map[opt.solver](denoiser)
    else:
        raise NotImplementedError

    return solver

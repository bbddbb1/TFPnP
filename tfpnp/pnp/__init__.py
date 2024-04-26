from .solver.base import PnPSolver
from .denoiser import UNetDenoiser2D, SCIUNetDenoiser, TvChambolleDenoiser


def create_denoiser(opt):
    print(f'[i] use denoiser: {opt.denoiser}')
    
    if opt.denoiser == 'unet':
        denoiser = UNetDenoiser2D()
    elif opt.denoiser == 'unet-sci':
        denoiser = SCIUNetDenoiser()
    elif opt.denoiser == 'tv_denoiser':
        denoiser = TvChambolleDenoiser()
    else:
        raise NotImplementedError

    return denoiser
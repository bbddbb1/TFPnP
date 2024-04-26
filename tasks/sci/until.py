import torch
import numpy as np



def A(x, Phi):
    '''
    Forward model of snapshot compressive imaging (SCI), where multiple coded
    frames are collapsed into a snapshot measurement.
    '''
    return torch.sum(x * Phi, dim=2)  # element-wise product


def At(y, Phi):
    '''
    Tanspose of the forward model.
    '''
    # (nrow, ncol, nmask) = Phi.shape
    # x = np.zeros((nrow, ncol, nmask))
    # for nt in range(nmask):
    #     x[:,:,nt] = np.multiply(y, Phi[:,:,nt])
    # return x
    if isinstance(y, torch.Tensor):
        return torch.mul(y.unsqueeze(2).repeat(1, 1, Phi.shape[2]), Phi)
    return np.multiply(np.repeat(y[:,:,np.newaxis],Phi.shape[2],axis=2), Phi)

def A_CHW(x, Phi):
    '''
    Forward model of snapshot compressive imaging (SCI), where multiple coded
    frames are collapsed into a snapshot measurement.
    '''
    return torch.sum(x * Phi, dim=1)  # element-wise product


def At_CHW(y, Phi):
    '''
    Tanspose of the forward model.
    '''
    # (nrow, ncol, nmask) = Phi.shape
    # x = np.zeros((nrow, ncol, nmask))
    # for nt in range(nmask):
    #     x[:,:,nt] = np.multiply(y, Phi[:,:,nt])
    # return x
    return torch.mul(y.unsqueeze(0).repeat(Phi.shape[0], 1, 1), Phi)


def At_BCHW(y, Phi):
    '''
    Tanspose of the forward model.
    '''
    # (nrow, ncol, nmask) = Phi.shape
    # x = np.zeros((nrow, ncol, nmask))
    # for nt in range(nmask):
    #     x[:,:,nt] = np.multiply(y, Phi[:,:,nt])
    # return x
    return torch.mul(y.unsqueeze(1).repeat(1, Phi.shape[1], 1, 1), Phi)


def shift_back(inputs,step):
    [row,col,nC] = inputs.shape
    for i in range(nC):
        if isinstance(inputs, torch.Tensor):
            inputs[:, :, i] = torch.roll(inputs[:, :, i], (-1) * step * i, dims=1)
        elif isinstance(inputs, np.ndarray):
            inputs[:, :, i] = np.roll(inputs[:, :, i], (-1)*step*i, axis=1)
    output = inputs[:, 0:col-step*(nC-1), :]
    return output


def shift_back_CHW(inputs,step):
    [nC, row,col] = inputs.shape
    for i in range(nC):
        if isinstance(inputs, torch.Tensor):
            inputs[i, :, :] = torch.roll(inputs[i, :, :], (-1) * step * i, dims=1)
        elif isinstance(inputs, np.ndarray):
            inputs[i, :, :] = np.roll(inputs[i, :, :], (-1)*step*i, axis=1)
    output = inputs[:, :, 0:col-step*(nC-1)]
    return output


def shift_back_bcwh(inputs,step):
    b, nC, row, col = inputs.shape
    output_list = []
    for i in range(b):
        res = shift_back_CHW(inputs[i], step)
        output_list.append(res)
    return torch.stack(output_list, dim=0)

def shift(inputs,step):
    [row,col,nC] = inputs.shape
    output = np.zeros((row, col+(nC-1)*step, nC))
    for i in range(nC):
        output[:,i*step:i*step+row,i] = inputs[:,:,i]
    return output


def shift_CHW(inputs, step):
    [nC, row, col] = inputs.shape
    if isinstance(inputs, torch.Tensor):
        output = torch.zeros((nC, row, col+(nC-1)*step), dtype=inputs.dtype, device=inputs.device)
    elif isinstance(inputs, np.ndarray):
        output = np.zeros((nC, row, col+(nC-1)*step))
    else:
        raise f"Unexpected inputs type"
    for i in range(nC):
        output[i:, :, i*step:i*step+row] = inputs[i, :, :]
    return output


def shift_bcwh(inputs, step):
    try:
        [b, nC, row, col] = inputs.shape
    except:
        pass
    output_list = []
    for i in range(b):
        res = shift_CHW(inputs[i], step)
        output_list.append(res)
    return torch.stack(output_list, dim=0)

from typing import Union

import pickle
import numpy as np
import torch
from torch.nn.functional import conv2d, conv3d

from deep_morpho.models import BiSE


#### Changes in original Deep_morpho library

# def _conv_fn_gen(ar, selem, **kwargs):
#     if ar.ndim == 2:
#         conv_layer = Conv2d(padding=((selem.shape[0]) // 2, (selem.shape[1]) // 2), bias=False, **kwargs)
#     else:
#         conv_layer = Conv3d(padding=((selem.shape[0]) // 2, (selem.shape[1]) // 2, 0), bias=False, **kwargs)
#     for param in conv_layer.parameters():
#         del param
#     return conv_layer._conv_forward


# def _old_erodila_conv(ar, selem, device):
#     conv_fn = _conv_fn_gen(ar, selem, padding_mode="zeros", in_channels=1, out_channels=1, kernel_size=selem.shape)
#     return conv_fn(
#         _format_for_conv(ar, device=device), _format_for_conv(selem, device=device), bias=torch.FloatTensor([0], device=device)
#     )

def _erodila_conv(ar, selem, device, convdim):
    if convdim == 2:
        while ar.ndim < 4:
            ar = ar.unsqueeze(0)
        while selem.ndim < 4:
            selem = selem.unsqueeze(0)
        return conv2d(ar.float().to(device), selem.float().to(device), padding=(selem.shape[-2] // 2, selem.shape[-1] // 2))
    else:
        while ar.ndim < 5:
            ar = ar.unsqueeze(0)
        while selem.ndim < 5:
            selem = selem.unsqueeze(0)
        return conv3d(ar.float().to(device), selem.float().to(device), padding=(selem.shape[-3] // 2, selem.shape[-2] // 2, selem.shape[-1] // 2))


def my_erosion(ar: np.ndarray | torch.Tensor, selem: np.ndarray | torch.Tensor, device: torch.device = "cpu", convdim : int = 2, return_numpy_array: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    Perform an erosion using torch conv2d and conv3d modules.
    For dimension details of tensors, see torch.nn.functionals.conv{2,3}d.

    Parameters
    ----------
    ar : np.ndarray | torch.Tensor
        Image to erode, with shape ((T,) H, W ), (Channel, (T,) H, W) or (MiniBatch, Channel, (T,) H, W) (T if `convdim = 3`).
    selem : np.ndarray | torch.Tensor
        Element S to erode `ar` with, with shape ((T,) H,W), (Channel//groups, (T,) H, W) or (OutChannels, Channel//groups, (T,) H, W) (T if `convdim = 3`).
    device : torch.device
        Device to send the `ar` and `selem` tensors.
    convdim : int = 2, in [2,3]
        Dimension of the convolution to perform.
    return_numpy_array : bool = False
        Convert the output in numpy.ndarray.

    Outputs
    -------
    torch.Tensor | np.ndarray
        Tensor of dimension (MiniBatch=1, Channel=1, (T,) H, W).

    Based on the implementation of T. AOUAD, erosion definition of P. LASCABETTES.
    """
    # torch_array = (_old_erodila_conv(ar, selem, device) == selem.sum()).squeeze((0,1))
    if not isinstance(ar, torch.Tensor):
        ar = torch.tensor(ar)
    if not isinstance(selem, torch.Tensor):
        selem = torch.tensor(selem)

    conv_results = _erodila_conv(ar, selem, device, convdim)
    if selem.shape[-1] %2 == 0:
        conv_results = conv_results[..., 1:]
    if selem.shape[-2] %2 == 0:
        conv_results = conv_results[..., 1:, :]
    if convdim == 3 and selem.shape[-3] %2 == 0:
        conv_results = conv_results[..., 1:, :, :]

    torch_array = (conv_results == selem.sum())

    if return_numpy_array:
        return torch_array.to("cpu").int().numpy()
    return torch_array


def correlation(ar: np.ndarray | torch.Tensor, selem: np.ndarray | torch.Tensor, device: torch.device = "cpu", convdim : int = 2, return_numpy_array: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    

    Parameters
    ----------
    ar : np.ndarray | torch.Tensor
        Image to erode, with shape ((T,) H, W ), (Channel, (T,) H, W) or (MiniBatch, Channel, (T,) H, W) (T if `convdim = 3`).
    selem : np.ndarray | torch.Tensor
        Element S to erode `ar` with, with shape ((T,) H,W), (Channel//groups, (T,) H, W) or (OutChannels, Channel//groups, (T,) H, W) (T if `convdim = 3`).
    device : torch.device
        Device to send the `ar` and `selem` tensors.
    convdim : int = 2, in [2,3]
        Dimension of the convolution to perform.
    return_numpy_array : bool = False
        Convert the output in numpy.ndarray.

    Outputs
    -------
    torch.Tensor | np.ndarray
        Tensor of dimension (MiniBatch=1, Channel=1, (T,) H, W).
    """
    # torch_array = (_old_erodila_conv(ar, selem, device) == selem.sum()).squeeze((0,1))
    if not isinstance(ar, torch.Tensor):
        ar = torch.tensor(ar)
    if not isinstance(selem, torch.Tensor):
        selem = torch.tensor(selem)

    conv_results = _erodila_conv(ar, selem, device, convdim)
    if selem.shape[-1] %2 == 0:
        conv_results = conv_results[..., 1:]
    if selem.shape[-2] %2 == 0:
        conv_results = conv_results[..., 1:, :]
    if convdim == 3 and selem.shape[-3] %2 == 0:
        conv_results = conv_results[..., 1:, :, :]

    torch_array = conv_results / selem.sum()

    if return_numpy_array:
        return torch_array.to("cpu").int().numpy()
    return torch_array


def my_dilatation(ar: np.ndarray, selem: np.ndarray, device: torch.device = "cpu", convdim: int = 2, return_numpy_array: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    Perform a dilatation using torch conv2d and conv3d modules.
    For dimension details of tensors, see torch.nn.functionals.conv{2,3}d.

    Parameters
    ----------
    ar : np.ndarray | torch.Tensor
        Image to dilate, with shape ((T,) H, W ), (Channel, (T,) H, W) or (MiniBatch, Channel, (T,) H, W) (T if `convdim = 3`).
    selem : np.ndarray | torch.Tensor
        Element S to dilate `ar` with, with shape ((T,) H,W), (Channel//groups, (T,) H, W) or (OutChannels, Channel//groups, (T,) H, W) (T if `convdim = 3`).
    device : torch.device
        Device to send the `ar` and `selem` tensors.
    convdim : int = 2, in [2,3]
        Dimension of the convolution to perform.
    return_numpy_array : bool = False
        Convert the output in numpy.ndarray.

    Outputs
    -------
    torch.Tensor | np.ndarray
        Tensor of dimension (MiniBatch=1, Channel=1, (T,) H, W).

    Based on the implementation of T. AOUAD.
    """
    if not isinstance(ar, torch.Tensor):
        ar = torch.tensor(ar)
    if not isinstance(selem, torch.Tensor):
        selem = torch.tensor(selem)
    
    flipselem = torch.flip(selem, (0,1))
    #torch_array = (_old_erodila_conv(ar, flipselem, device) > 0).squeeze((0,1))
    conv_results = _erodila_conv(ar, flipselem, device, convdim)
    if selem.shape[-1] %2 == 0:
        conv_results = conv_results[..., :-1]
    if selem.shape[-2] %2 == 0:
        conv_results = conv_results[..., :-1, :]
    if convdim == 3 and selem.shape[-3] %2 == 0:
        conv_results = conv_results[..., :-1, :, :]

    torch_array = (conv_results > 0)
    

    if return_numpy_array:
        return torch_array.to("cpu").int().numpy()
    return torch_array


class BiSERectified(BiSE):

    @property
    def learned_selem(self):
        "Correct the dilatation mirrorization."
        if self._learned_selem is None:
            return None
        
        out = np.zeros(self._learned_selem.shape)
        for i, op in enumerate(self._learned_operation):
            if op == 1:
                out[i] = np.flip(self._learned_selem[i], (-2,-1))
            else:
                out[i] = self._learned_selem[i]
        return out
    
    @property
    def closest_selem(self):
        "Correct the dilatation mirrorization."
        if self._closest_selem is None:
            return None
        
        out = np.zeros(self._closest_selem.shape)
        for i, op in enumerate(self._closest_operation):
            if op == 1:
                out[i] = np.flip(self._closest_selem[i], (-2,-1))
            else:
                out[i] = self._closest_selem[i]
        return out
    

def save_data_and_kwargs(path:str, data, **kwargs):
    """An easy way to save arguments.
    
    
    Read syntax :
    with open("file.pkl", 'rb') as file:
        data, kwargs = pickle.load(file)
    """
    with open(path, "wb") as file:
        pickle.dump([data, kwargs], file)
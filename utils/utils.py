import numpy as np
import torch
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from functools import partial
import random

def rsshow(I, scale=0.005):
    low, high = np.quantile(I, [scale, 1 - scale])
    I[I > high] = high
    I[I < low] = low
    I = (I - low) / (high - low)
    return I

def data_augmentation(image, mode=None):
    """
    Args:
        image: np.ndarray, shape: C X H X W
    """
    axes = (-2, -1)
    flipud = lambda x: x[:, ::-1, :]

    if mode is None:
        mode = random.randint(0, 8)
    if mode == 0:
        # original
        image = image
    elif mode == 1:
        # flip up and down
        image = flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = np.rot90(image, axes=axes)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image, axes=axes)
        image = flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = np.rot90(image, k=2, axes=axes)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes=axes)
        image = flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = np.rot90(image, k=3, axes=axes)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=axes)
        image = flipud(image)

    return np.ascontiguousarray(image)

class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-3]
        bwindex = []
        for ch in range(C):
            x = torch.squeeze(X[...,ch,:,:].data).cpu().numpy()
            y = torch.squeeze(Y[...,ch,:,:].data).cpu().numpy()
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex

cal_bwssim = Bandwise(partial(compare_ssim, data_range=1))
cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=1))


def cal_sam(X, Y, eps=1e-8):
    X = torch.squeeze(X.data).cpu().numpy()
    Y = torch.squeeze(Y.data).cpu().numpy()
    tmp = np.sum(X*Y, axis=0) / (np.sqrt(np.sum(X**2, axis=0)) * np.sqrt(np.sum(Y**2, axis=0)) + eps)
    return np.mean(np.real(np.arccos(tmp)))

def cal_ergas(X, Y):
    if len(X.shape) == 4:
        X = X[None, ...]
        Y = Y[None, ...]
    # Metric = iv.spectra_metric(Y[0, 0, ...].permute(1,2,0).detach().cpu().numpy(), X[0, 0, ...].permute(1,2,0).detach().cpu().numpy(),
    #                            scale=1)
    # ERGAS = Metric.ERGAS()

    ergas = 0
    for i in range(X.size(2)):
        ergas = ergas + torch.nn.functional.mse_loss(X[:,:, i, ...], Y[:,:, i, ...]) / torch.mean(X[:,:, i, ...]) #** 2
    ergas = 100 * torch.sqrt(ergas / X.size(2))
    ergas = ergas.item()
    return ergas

def cal_ssim(X, Y):
    ssim = np.mean(cal_bwssim(X, Y))
    return ssim

def MSIQA(X, Y):
    psnr = np.mean(cal_bwpsnr(X, Y))
    ssim = np.mean(cal_bwssim(X, Y))
    sam = cal_sam(X, Y)
    ergas = cal_ergas(X, Y)
    return psnr, ssim, sam, ergas


import torch
from utils import *

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x, nrow=8)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

# Define function to compute peak signal to noise ratio (PSNR)
from math import log10, sqrt 

def PSNR(real, reconstructed): 
    mse = np.mean((real - reconstructed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

from torchmetrics.functional.image import structural_similarity_index_measure
def normalize(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    x_im = x.detach().cpu().clip(0, 1) * 255
    x_im = np.array(x_im).astype(np.uint8)
    return x_im

# Define libraries
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from diffusers import UNet2DModel

pixel_size = 256
channels = 4
image_size = pixel_size//8

class cont_cond_unet_diffusion_model(nn.Module):
    def __init__(self):
        super().__init__()
        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=image_size,           # the target image resolution
            in_channels=channels + 1, # Additional input channels for class cond.
            out_channels=channels,           # the number of output channels
            layers_per_block=2,       # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
            down_block_types=( 
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D", 
                "DownBlock2D", 
                "DownBlock2D", 
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ), 
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D"  
            ),
        )

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, y):
        # Shape of x:
        bs, ch, w, h = x.shape
        class_cond = y.view(-1,1)
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        net_input = torch.cat((x, class_cond), 1)
        # Feed this to the unet alongside the timestep and return the prediction
        return self.model(net_input, t).sample 

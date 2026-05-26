# Import libraries
import torch
import numpy as np
import random
import copy
import gc
import sys
sys.path.append("..")
from models import *
import matplotlib.pyplot as plt
from diffusers import LMSDiscreteScheduler, DDIMScheduler, AutoencoderKL
from utils import *
import timeit
from tqdm.auto import tqdm
from ddim import *

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
# To the GPU we go!
vae = vae.to(device)

# The noise scheduler
noise_scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)

# data loader
data_filename1 = '../fluid_256x256.npy' # Contains {20<=Re<=40 and 100<=Re<=120 with r=0.05}, {R=20 with 0.001=<r<0.05 and R=120 with 0.05<r=<0.01}  
data_filename2 = '../fluid_test_Re_256x256.npy' # Contains dynamics of Re=45,55,65,75,85,95
v_images_1 = np.load(data_filename1).reshape(82,3000,256,256,3)
v_images_2 = np.load(data_filename2).reshape(6,3000,256,256,3)
idx_time = list(range(0,3000,30)) # select 100 time points
v_images = np.concatenate((v_images_1[:,idx_time],v_images_2[:,idx_time]))
print(v_images.shape)

N = v_images_1.shape[0] + v_images_2.shape[0]
T = len(idx_time)
v_images_all = v_images.reshape(N*T,256,256,3).transpose((0,3,1,2))

labels_list = np.linspace(0.0005,1.5,3000)
labels_all = np.tile(labels_list[idx_time], N)
labels_train_raw = labels_all

images_train = v_images_all
print(images_train.shape)

# normalize labels_train_raw
print("\n Range of unnormalized labels: ({},{})".format(np.min(labels_train_raw), np.max(labels_train_raw)))

labels_train = (labels_train_raw - np.min(labels_train_raw)) / (np.max(labels_train_raw)-np.min(labels_train_raw))
print("\n Range of normalized labels: ({},{})".format(np.min(labels_train), np.max(labels_train))) 

niters = 40000
save_models_folder = '../output/saved_models'
Filename_LDM = save_models_folder + '/CcLDM_checkpoint_intrain/CcLDM_checkpoint_niters_{}.pth'.format(niters)
print("Loading pre-trained continuous conditional latent diffusion model >>>")

checkpoint = torch.load(Filename_LDM, map_location=device,weights_only=True)
netLDM = cont_cond_unet_diffusion_model().to(device)
netLDM = nn.DataParallel(netLDM)

# Adjust checkpoint keys if necessary
checkpoint_state_dict = checkpoint['netLDM_state_dict']
adjusted_state_dict = {}

for key, value in checkpoint_state_dict.items():
    # Replace naming inconsistencies
    new_key = key.replace("query", "to_q").replace("key", "to_k").replace("value", "to_v").replace("proj_attn", "to_out.0")
    adjusted_state_dict[new_key] = value

# Load the adjusted state dict
netLDM.load_state_dict(adjusted_state_dict, strict=True)

def torch_images(X, Y, batch_size):
    trainset = IMGs_dataset(X, Y, normalize=True)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    dataloader_iter = iter(train_dataloader)
    b_train_images, b_train_labels = next(dataloader_iter)
    assert batch_size == b_train_images.shape[0]
    b_train_images = b_train_images.type(torch.float).to(device)
    b_train_labels = b_train_labels.type(torch.float).to(device)
    with torch.no_grad():
        latents = vae.encode(b_train_images).latent_dist.sample()
    latent_b_train_images = latents * 0.18215  
    return b_train_images, b_train_labels, latent_b_train_images

from ddim import *
def DDIM_latents(Z, Y):
    start_step = 0
    inf_steps = 1000
    inverted_latents = invert(Z, num_inference_steps=inf_steps, labels=Y,
                      noise_scheduler=noise_scheduler, netLDM=netLDM, device=device)
    return inverted_latents[-1]

print('number of images',len(images_train))

i=0
bs = 44
real_images = []
noisy_latent_images = []
for idx in range(0,len(images_train),bs):
    # Get training images
    X, Y, Z = torch_images(images_train[idx:idx+bs], labels_train[idx:idx+bs], bs)
    i=i+1
    print(i)
    real_images.append(X)
    noisy_latent_images.append(DDIM_latents(Z, Y))
    
latent_space = torch.stack(noisy_latent_images).detach().cpu().numpy()
print(latent_space.shape)
Z_data = latent_space.reshape(len(images_train),4*32*32)

real_data = torch.stack(real_images).detach().cpu().numpy()
print(real_data.shape)

X_data = real_data.reshape(len(images_train),3,256,256)

np.save('X_class.npy', X_data)
np.save('Z_class.npy', Z_data)
np.save('norm_labels.npy', labels_train)
np.save('raw_labels.npy', labels_train_raw)
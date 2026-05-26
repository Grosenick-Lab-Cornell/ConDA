import torch
import numpy as np
import random
from models import *
import matplotlib.pyplot as plt
from diffusers import LMSDiscreteScheduler, DDIMScheduler, AutoencoderKL
from utils import *
import timeit
import h5py
import os

from interpolation_baselines.diff_space import *
from evaluation import *
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

# Load dataset
f = h5py.File('TScells_chunked_256x256_3D_normtif_2col_timeidx_float32_413968.h5','r')

# Calcium signals images
images = f['tif_arr']
# Time point of data
labels_raw = f['y_time'][:]

q1 = 59
q2 = 19999

print("\n Range of unnormalized labels: ({},{})".format(np.min(labels_raw), np.max(labels_raw))) 
# Normalize the labels
labels = np.empty(labels_raw.shape)
labels[:,0] = labels_raw[:,0] / q1
labels[:,1] = labels_raw[:,1] / q2
print("\n Range of normalized labels: ({},{})".format(np.min(labels), np.max(labels)))

# M = int(sys.argv[1]) vary from 0 to 59
M = 20 # M=20 or M=8
idx_y1 = np.where(labels_raw[:,0] == M)[0]
labels_raw1 = labels_raw[idx_y1]
print(len(labels_raw1))

# Load pre-trained model
niters = 50000
save_models_folder = 'output/saved_models'

Filename_LDM = save_models_folder + '/CcLDM_checkpoint_intrain/CcLDM_checkpoint_niters_{}.pth'.format(niters)
print("Loading pre-trained continuous conditional latent diffusion model >>>")
checkpoint = torch.load(Filename_LDM, map_location=device, weights_only=True)
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
    #assert batch_size == b_train_images.shape[0]
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

from Rex_RK4 import *
def Rex_latents(Z, Y):
    inf_steps = 8
    noise_scheduler.set_timesteps(inf_steps, device=device)  # e.g., torch.linspace(0., T, steps+1) or log-sigma grid
    t_grid = noise_scheduler.timesteps.to(torch.float32)
    # Rex prefers time increasing (0 → 1), so flip them:
    t_grid = torch.flip(t_grid,dims=[0])
    schedule = DDIMPfODEAdapter(noise_scheduler, t_grid)
    pf_ode = PFODE(model=netLDM, schedule=schedule, pred_type="eps", cond=Y)
    z_T   = rex_encode(pf_ode, t_grid, Z)        # data -> latent (ODE integrate 0→T)
    return z_T

images = images[idx_y1]
labels = labels[idx_y1]
print('number of images',len(images))
i=0
bs = 50
real_images = []
noisy_latent_images = []

for idx in range(0,len(images),bs):
    end_idx = min(idx + bs, len(images))
    # Get training images
    X, Y, Z = torch_images(images[idx:end_idx], labels[idx:end_idx], bs)
    i=i+1
    print(i)
    real_images.append(X)
    noisy_latent_images.append(DDIM_latents(Z, Y)) # Use Rex_latents for Rex-RK4
    
#latent_space = torch.stack(noisy_latent_images).detach().cpu().numpy()
latent_space = torch.cat(noisy_latent_images, dim=0).detach().cpu().numpy()
print(latent_space.shape)
Z_data = latent_space.reshape(len(images),4*32*32)

real_data = torch.cat(real_images, dim=0).detach().cpu().numpy()
print(real_data.shape)

X_data = real_data.reshape(len(images),3,256,256)

os.makedirs("data", exist_ok=True)

np.save('data/X_data_{}.npy'.format(M), X_data)
np.save('data/ddim_latents_{}.npy'.format(M), Z_data)
np.save('data/norm_labels_{}.npy'.format(M), labels)
np.save('data/raw_labels_{}.npy'.format(M), labels_raw)
    
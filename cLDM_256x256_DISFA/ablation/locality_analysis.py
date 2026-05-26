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
from evaluation import *
from ddim import *
from Cebra import cebra_model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
# To the GPU we go!
vae = vae.to(device)


# The noise scheduler
noise_scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)

# Load Data
# Load Data
subject = 'SN025' 
Z_data = np.load('../data/ddim_latents_{}.npy'.format(subject))
au_labels = np.load('../data/norm_labels_{}.npy'.format(subject))
labels_raw = np.load('../data/raw_labels_{}.npy'.format(subject))
cebra_model.fit(Z_data,au_labels)

import cebra

# Total frames and duration
num_frames = len(Z_data)
total_duration_sec = 4 * 60  # 4 minutes = 240 seconds

# Linearly spaced time points
time_seconds_raw = np.linspace(0, total_duration_sec, num_frames)
time_seconds_normalized = time_seconds_raw/total_duration_sec
labels = time_seconds_normalized

number_list = list(range(4840))
train_block_size = 44
test_block_size = 11
num_blocks = 88

train_blocks = [number_list[i * (train_block_size+test_block_size)  : (i + 1) * train_block_size + test_block_size*i] 
                for i in range(num_blocks)]
test_blocks = [number_list[(i + 1) * train_block_size + test_block_size*i : (i+1)*(train_block_size + test_block_size)]
               for i in range(num_blocks)]  
idx_train = []
idx_test = []
for j in range(num_blocks):
    idx_train = idx_train + train_blocks[j]
    idx_test = idx_test + test_blocks[j]

print("Train:", len(idx_train), "Test:", len(idx_test))

C = cebra_model.transform(Z_data)

knn_model = cebra.KNNDecoder(n_neighbors=1, metric="euclidean")
knn_model.fit(C, Z_data)

def rmse_pointwise(X, Y):
    return np.sqrt(np.mean(np.sum((X - Y)**2, axis=1)))

# perturb C latents
distances = []
rmse_values = []
delta_list = []
Z_pred_values = []

deltas = list(range(6))
scale = 0.6

seq_idx = list(range(0,len(idx_test),18))


c_seq = C[idx_test][seq_idx]   # (T, 3)
z_seq = Z_data[idx_test][seq_idx]   # (T, Dz)

T = len(c_seq)

for delta in deltas:
    for t in range(T):
        #if t + delta >= T:
        #    continue
        c_anchor = c_seq[t]              # (3,)
        noise = np.random.randn(3)
        noise = noise / np.linalg.norm(noise)
        alpha = scale * delta
        c_edited = c_anchor + alpha * noise
        z_anchor = z_seq[t] 
        
        # Distance in 3D ConDA space
        d_c = np.linalg.norm(c_edited - c_anchor)
            
        # Decode target point from C to Z
        z_pred = knn_model.predict(c_edited.reshape(1, -1))  # (1, Dz)

        # Compare to true target latent
        err = rmse_pointwise(z_pred, z_anchor.reshape(1, -1))

        #print(f"delta={delta:3d}, distance={d_c:.6f}, rmse={err:.6f}")

        distances.append(d_c)
        rmse_values.append(err)
        delta_list.append(delta)
        Z_pred_values.append(z_pred)

idx_intp = seq_idx
import pandas as pd
distances = np.array(distances).reshape(len(deltas),len(idx_intp))
rmse_values = np.array(rmse_values).reshape(len(deltas),len(idx_intp))
delta_list = np.array(delta_list).reshape(len(deltas),len(idx_intp))

rows = []
for d in range(len(deltas)):
    rows.append({
        "delta": deltas[d],
        "mean_distance": distances[d].mean(),
        #"std_distance": distances[mask].std(),
        "mean_rmse": rmse_values[d].mean(),
        #"std_rmse": rmse_values[mask].std(),
        "count": len(idx_intp)
    })
df_delta = pd.DataFrame(rows)
print(df_delta)

niters = 100000
save_models_folder = '../output/saved_models'

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

Z_pred_values = np.array(Z_pred_values).reshape(len(deltas),len(idx_intp),4,32,32)
ddim_test = torch.tensor(Z_pred_values).type(torch.float).to(device)

test_labels = torch.tensor(au_labels[idx_test]).type(torch.float).to(device)

X_data = np.load('../data/X_data_{}.npy'.format(subject))[idx_test]
real_images = torch.tensor(X_data[idx_intp]).type(torch.float) 

decoded_images = []
inf_steps= 600
for d in range(len(deltas)):
    ddim_dec = sample(start_latents=ddim_test[d], num_inference_steps=inf_steps, 
                           labels=test_labels[idx_intp], noise_scheduler=noise_scheduler, netLDM=netLDM, 
                           start_step=0, device=device)
    ddim_dec_ = (1 / 0.18215) * ddim_dec[-1]
    with torch.no_grad():
        ddim_sample_dec = vae.decode(ddim_dec_).sample 
  
    print(ddim_sample_dec.shape)
    
    psnr = []
    SSIM = []

    for i in range(len(idx_intp)):
        psnr.append(PSNR(normalize(real_images[i]), normalize(ddim_sample_dec[i])))
        SSIM.append(structural_similarity_index_measure(ddim_sample_dec[i:i+1].to(device),
                                                           real_images[i:i+1].to(device)).item()) 
    print(f"PSNR: Mean ± Std: {np.mean(psnr):.2f} ± {np.std(psnr):.2f}")
    print(f"SSIM: Mean ± Std: {np.mean(SSIM):.2f} ± {np.std(SSIM):.2f}")  

    decoded_images.append(ddim_sample_dec)  
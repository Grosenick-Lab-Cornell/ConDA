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
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
# To the GPU we go!
vae = vae.to(device)


# The noise scheduler
noise_scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)

# Load Data
M = 20

labels_raw = np.load('../data/raw_labels_{}.npy'.format(M))
Z_data = np.load('../data/ddim_latents_{}.npy'.format(M))
ca_labels = np.load('../data/norm_labels_{}.npy'.format(M))

labels = ca_labels[:,1]
num_frames = len(labels)
cebra_model.fit(Z_data,ca_labels)
C_data = cebra_model.transform(Z_data)

number_list = list(range(4800))
train_block_size = 41
test_block_size = 7
num_blocks = 100
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

cebra_model.fit(Z_data[idx_train],ca_labels[idx_train])
C_train = cebra_model.transform(Z_data[idx_train])
C_test = cebra_model.transform(Z_data[idx_test])

# kNN 
import cebra
knn_model = cebra.KNNDecoder(metric="euclidean")
knn_model.fit(C_train, Z_data[idx_train])
knn_pred = knn_model.predict(C_test)

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ---- 1. MLP Decoder: (d) -> (4096) ---- #

class MLPDecoder(nn.Module):
    def __init__(self,
                 in_dim=3,
                 out_dim=4*32*32,
                 hidden_dims=(256, 256),
                 dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---- 2. Training function ---- #

def train_decoder(
    C_emb,              # (T, 8) numpy or torch
    X_fields,           # (T, 4*32*32) or (T, 4, 32, 32)
    hidden_dims=(256, 256),
    dropout=0.0,
    batch_size=128,
    lr=1e-3,
    weight_decay=0.0,
    num_epochs=100,
    val_split=0.2,
    device=None,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure numpy -> torch
    if isinstance(C_emb, np.ndarray):
        C_emb = torch.from_numpy(C_emb).float()
    if isinstance(X_fields, np.ndarray):
        X_fields = torch.from_numpy(X_fields).float()

    # If X_fields is (T, 4, 32, 32), flatten to (T, 4096)
    if X_fields.ndim == 4:
        T, C, H, W = X_fields.shape
        assert (C, H, W) == (4, 32, 32)
        X_flat = X_fields.view(T, -1)
    else:
        X_flat = X_fields   # assume already (T, 4096)

    T_total, in_dim = C_emb.shape
    out_dim = X_flat.shape[1]

    # Train/val split
    n_val = int(T_total * val_split)
    n_train = T_total - n_val

    C_train, C_val = torch.split(C_emb, [n_train, n_val], dim=0)
    X_train, X_val = torch.split(X_flat, [n_train, n_val], dim=0)

    train_ds = TensorDataset(C_train, X_train)
    val_ds   = TensorDataset(C_val, X_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model, optimizer, loss
    model = MLPDecoder(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        for C_b, X_b in train_loader:
            C_b = C_b.to(device)
            X_b = X_b.to(device)

            optimizer.zero_grad()
            X_hat = model(C_b)
            loss = criterion(X_hat, X_b)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * C_b.size(0)

        train_loss = train_loss_sum / n_train

        # ---- Val ----
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for C_b, X_b in val_loader:
                C_b = C_b.to(device)
                X_b = X_b.to(device)
                X_hat = model(C_b)
                loss = criterion(X_hat, X_b)
                val_loss_sum += loss.item() * C_b.size(0)

        val_loss = val_loss_sum / n_val
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:03d}: "
              f"train {train_loss:.4e} | val {val_loss:.4e}")

    return model, (train_losses, val_losses)


configs = [
    {"hidden_dims": (256, 256), "dropout": 0.0, "lr": 1e-3, "weight_decay": 0.0},
    {"hidden_dims": (512, 512), "dropout": 0.05, "lr": 1e-3, "weight_decay": 1e-5},
    {"hidden_dims": (512, 1024, 512), "dropout": 0.05, "lr": 1e-3, "weight_decay": 1e-5},
    {"hidden_dims": (512, 1024, 1024, 512), "dropout": 0.05, "lr": 5e-4, "weight_decay": 1e-5},
    {"hidden_dims": (1024, 1024, 1024), "dropout": 0.1, "lr": 5e-4, "weight_decay": 1e-4},
]

best_model = None
best_val = float("inf")
best_config = None

for cfg in configs:
    print("\nTesting config:", cfg)

    model, (train_hist, val_hist) = train_decoder(
        C_emb=C_train,
        X_fields=Z_data[idx_train],
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
        batch_size=256,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        num_epochs=300,
        val_split=0.2,
    )

    final_val = min(val_hist)

    if final_val < best_val:
        best_val = final_val
        best_model = model
        best_config = cfg

print("Best config:", best_config)
print("Best validation loss:", best_val)

# Reconstruct for all embeddings
device = next(best_model.parameters()).device
with torch.no_grad():
    C_tensor = torch.from_numpy(C_test).float().to(device)
    mlp_pred = model(C_tensor).cpu().numpy()    # (T, 4096)

niters = 50000
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

mlp_tensor = torch.tensor(mlp_pred.reshape(len(idx_test),4, 32, 32)).type(torch.float).to(device)
knn_tensor = torch.tensor(knn_pred.reshape(len(idx_test),4, 32, 32)).type(torch.float).to(device)
test_labels = torch.tensor(ca_labels[idx_test]).type(torch.float).to(device)

inf_steps = 1000
i = list(range(0,len(idx_test),14))
mlp_sample = sample(start_latents=mlp_tensor[i], num_inference_steps=inf_steps, 
                           labels=test_labels[i], noise_scheduler=noise_scheduler, netLDM=netLDM, 
                           start_step=0, device=device)
mlp_sample = (1 / 0.18215) * mlp_sample[-1]
with torch.no_grad():
    x_mlp = vae.decode(mlp_sample).sample 

knn_sample = sample(start_latents=knn_tensor[i], num_inference_steps=inf_steps, 
                           labels=test_labels[i], noise_scheduler=noise_scheduler, netLDM=netLDM, 
                           start_step=0, device=device)
knn_sample = (1 / 0.18215) * knn_sample[-1]
with torch.no_grad():
    x_knn = vae.decode(knn_sample).sample 


X_data = np.load('../data/X_data_{}.npy'.format(M))[idx_test]
real_images = torch.tensor(X_data[i]).type(torch.float) 

# Evaluation
from evaluation import *
print('PSNR and SSIM')

# For computing PSNR and SSIM
psnr_knn = PSNR(normalize(real_images), normalize(x_knn))
ssim_knn = structural_similarity_index_measure(x_knn,real_images.to(device)).item()
print('kNN:',psnr_knn,ssim_knn)  

psnr_mlp = PSNR(normalize(real_images), normalize(x_mlp))
ssim_mlp = structural_similarity_index_measure(x_mlp,real_images.to(device)).item()
print('MLP:',psnr_mlp,ssim_mlp)

print('RMSE')
# Trajectories are sampled at the same times (pointwise L2)
def rmse_pointwise(X, Y):  # X,Y: (T,dim) with same T
    return np.sqrt(np.mean(np.sum((X - Y)**2, axis=1)))  

rmse_knn = rmse_pointwise(Z_data[idx_test], knn_pred)
rmse_mlp = rmse_pointwise(Z_data[idx_test], mlp_pred)
print('kNN:',rmse_knn, 'MLP:',rmse_mlp)     

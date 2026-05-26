import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
from tqdm.auto import tqdm
import os
import timeit
from diffusers import LMSDiscreteScheduler, AutoencoderKL
from matplotlib import pyplot as plt
from utils import *
from opts import parse_opts

''' Settings '''
args = parse_opts()
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
# To the GPU we go!
vae = vae.to(device)

# some parameters in opts
niters = args.niters
resume_niters = args.resume_niters
lr = args.lr
save_niters_freq = args.save_niters_freq
batch_size = args.batch_size
v_batch_size = batch_size

# The noise scheduler
noise_scheduler = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)

def train_CcLDM(train_images, train_labels, val_images, val_labels, netLDM, save_images_folder, save_models_folder = None):

    '''  Note that train_images are not normalized to [-1,1]  '''
    netLDM = netLDM.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(netLDM.parameters(), lr=lr, betas=(0.5, 0.999))
    
    trainset = IMGs_dataset(train_images, train_labels, normalize=True)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    v_trainset = IMGs_dataset(val_images, val_labels, normalize=True)
    val_dataloader = torch.utils.data.DataLoader(v_trainset, batch_size=v_batch_size, shuffle=True)
    
    if save_models_folder is not None and resume_niters>0:
        save_file = save_models_folder + "/CcLDM_checkpoint_intrain/CcLDM_checkpoint_niters_{}.pth".format(resume_niters)
        checkpoint = torch.load(save_file)
        netLDM.load_state_dict(checkpoint['netLDM_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    
    batch_idx = 0
    dataloader_iter = iter(train_dataloader)
    
    v_batch_idx = 0
    v_dataloader_iter = iter(val_dataloader)
    
    fhist = open('output/loss/loss_CcLDM.dat','w')

    start_time = timeit.default_timer()
    for niter in range(resume_niters, niters):
       
       ############ TRAINING #####################
        if batch_idx+1 == len(train_dataloader):
            dataloader_iter = iter(train_dataloader)
            batch_idx = 0
            
        batch_real_images, batch_target_labels = next(dataloader_iter)
        assert batch_size == batch_real_images.shape[0]
        batch_real_images = batch_real_images.type(torch.float).to(device)
        batch_target_labels = batch_target_labels.type(torch.float).to(device)
  
        '''  Train Unet to denoise images  '''
        
        netLDM.train()
        
        # Convert images to latent space
        with torch.no_grad():
            latents = vae.encode(batch_real_images).latent_dist.sample()
            latent_batch_real_images = latents * 0.18215
        
        
        # Sample noise to add to the images
        noise = torch.randn(latent_batch_real_images.shape).to(device)
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (batch_size,)).long().to(device)
        
        # Add noise to the clean training images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(latent_batch_real_images, noise, timesteps).to(device)
        
        # Get the model prediction
        noise_pred = netLDM(noisy_images, timesteps, batch_target_labels) # Note that we pass in the labels y
        
        # Calculate the loss
        loss = criterion(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_idx+=1
        
        #############FOR VALIDATION ###############
        if v_batch_idx+1 == len(val_dataloader):
            v_dataloader_iter = iter(val_dataloader)
            v_batch_idx = 0    
        
        # validation images
        v_batch_real_images, v_batch_target_labels = next(v_dataloader_iter)
        assert v_batch_size == v_batch_real_images.shape[0]
        v_batch_real_images = v_batch_real_images.type(torch.float).to(device)
        v_batch_target_labels = v_batch_target_labels.type(torch.float).to(device)
        
        netLDM.eval()
        
        # Convert images to latent space
        with torch.no_grad():
            v_latents = vae.encode(v_batch_real_images).latent_dist.sample()
            latent_v_batch_real_images = latents * 0.18215
        
        
        # Sample noise to add to the test images
        v_noise = torch.randn(latent_v_batch_real_images.shape).to(device)
        
        # Sample a random timestep for each image
        v_timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (batch_size,)).long().to(device)
        
        # Add noise to the clean training images according to the noise magnitude at each timestep
        v_noisy_images = noise_scheduler.add_noise(latent_v_batch_real_images, v_noise, v_timesteps).to(device)
        
        # Get the model prediction
        v_noise_pred = netLDM(v_noisy_images, v_timesteps, v_batch_target_labels) # Note that we pass in the labels y        
                    
        # Calculate the loss
        v_loss = criterion(v_noise_pred, v_noise)
      
        v_batch_idx+=1
        #############END VALIDATION ###############
        
        if (niter+1)%1 == 0:
            print ("CcLDM: [Iter %d/%d] [loss: %.4f] [v_loss: %.4f] [Time: %.4f]" % (niter+1, niters, loss.item(), v_loss.item(), timeit.default_timer()-start_time))
            fhist.write(str(niter+1)+" "+str(loss.item())+" "+str(v_loss.item())+"\n")
            
        if save_models_folder is not None and ((niter+1) % save_niters_freq == 0 or (niter+1) == niters):
            save_file = save_models_folder + "/CcLDM_checkpoint_intrain/CcLDM_checkpoint_niters_{}.pth".format(niter+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'netLDM_state_dict': netLDM.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
            
            # Save some 8 training images in 2000 iterations
            save_image(batch_real_images[:8,:,:,:].detach(), 'output/clean_images/CcLDM_clean/{}.png'.format(niter+1), nrow=8, normalize=True)
            
            # Sampling: Random starting point (8 random images):
            latent_sample = torch.randn(8, 4, 32, 32).to(device)
            latent_sample = latent_sample * noise_scheduler.init_noise_sigma
            for i, t in enumerate(noise_scheduler.timesteps):
                # Get model pred
                with torch.no_grad():
                    residual = netLDM.module(latent_sample, torch.tensor([t], device=device), batch_target_labels[:8])
                # Update sample with step
                #latent_sample = noise_scheduler.step(residual, t, latent_sample).pred_original_sample 
                latent_sample = noise_scheduler.step(residual, t, latent_sample).prev_sample
            
            decoder = vae.to(device) 
            latent_sample = (1 / 0.18215) * latent_sample
            with torch.no_grad():
                decoded_sample = decoder.decode(latent_sample).sample
            save_image(decoded_sample.detach().cpu(), save_images_folder +'/{}.png'.format(niter+1), nrow=8, normalize=True)
            
    #end for niter
    return netLDM

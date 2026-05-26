import torch
from tqdm.auto import tqdm

@torch.no_grad()
def invert(start_latents, labels, noise_scheduler, netLDM, device, num_inference_steps=80):
  
    # latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(noise_scheduler.timesteps).long()

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps-1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1: continue
        
        t = timesteps[i]

        # predict the noise residual
        noise_pred = netLDM.module(latents, t, labels)
        #noise_pred = netLDM(latents, t)

        current_t = t 
        next_t = t+1 
        alpha_t = noise_scheduler.alphas_cumprod[current_t]
        alpha_t_next = noise_scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of 
        #x(t-1) (current latents)
        beta_t = 1 - alpha_t
        latents = (latents - beta_t ** 0.5 * noise_pred) / alpha_t ** 0.5
        next_sample_direction = (1 - alpha_t_next) ** 0.5 * noise_pred
        latents = alpha_t_next ** 0.5 * latents + next_sample_direction
        
        # Store
        intermediate_latents.append(latents)
            
    return intermediate_latents

#Define DDIM sampling (from inverted latents z* to z_0)
# Sample function (regular DDIM) (invert noisy latents back to original latents)
@torch.no_grad()
def sample(start_latents, labels, noise_scheduler, netLDM, device, start_step=0, num_inference_steps=30):
  
    # Set num inference steps
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    latents_list = []
    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 32, 32, device=device)
        start_latents *= noise_scheduler.init_noise_sigma

    latents = start_latents.clone()
    timesteps_sample = noise_scheduler.timesteps.long()
    for i in tqdm(range(start_step, num_inference_steps-1)):
    
        t = timesteps_sample[i]

        # predict the noise residual
        noise_pred = netLDM.module(latents, t, labels)
        #noise_pred = netLDM(latents, t)
        
        prev_t = t-1
        alpha_t = noise_scheduler.alphas_cumprod[t]
        alpha_t_prev = noise_scheduler.alphas_cumprod[prev_t]
        beta_t = 1 - alpha_t
        pred_original_sample = (latents - beta_t ** 0.5 * noise_pred) / alpha_t ** 0.5
        pred_sample_direction = (1 - alpha_t_prev) ** 0.5 * noise_pred
        prev_sample = alpha_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        latents = prev_sample

        # Store
        latents_list.append(latents)
    return latents_list


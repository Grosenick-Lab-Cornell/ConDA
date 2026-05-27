import torch
from tqdm.auto import tqdm

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## Inversion (from latents to noisy latents)
@torch.no_grad()
def invert(start_latents, labels, ema_denoiser_avg, scheduler, cfg, num_inference_steps=80):
  
    # latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    scheduler.set_timesteps(cfg.denoiser_model.num_train_timesteps)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(scheduler.timesteps).long()

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps-1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1: continue
        
        t = timesteps[i]
        
        # predict the noise residual
        noise_pred = ema_denoiser_avg(
            torch.cat([latents, labels], dim=1),
            torch.tensor([t] * len(latents)).to(device).long(),
        )
        noise_pred = noise_pred[:, :-2]

        current_t = t #max(0, t.item()-1)#max(0, t.item() - (1000//num_inference_steps)) #t
        next_t = t+1 #min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = scheduler.alphas_cumprod[current_t]
        alpha_t_next = scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        #latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred
        beta_t = 1 - alpha_t
        latents = (latents - beta_t ** 0.5 * noise_pred) / alpha_t ** 0.5
        next_sample_direction = (1 - alpha_t_next) ** 0.5 * noise_pred
        latents = alpha_t_next ** 0.5 * latents + next_sample_direction
        
        # Store
        latents
            
    return latents

# Sample function (regular DDIM) (invert noisy latents back to original latents)
@torch.no_grad()
def sample(labels, scheduler, ema_denoiser_avg, cfg, start_step=0, start_latents=None, num_inference_steps=30):
  
    # Set num inference steps
    scheduler.set_timesteps(cfg.denoiser_model.num_train_timesteps)
    latents_list = []
    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 32, 32, device=device)
        start_latents *= noise_scheduler.init_noise_sigma

    latents = start_latents.clone()
    timesteps_sample = scheduler.timesteps.long()
    for i in tqdm(range(start_step, num_inference_steps-1)):
    
        #t = noise_scheduler.timesteps[i]
        t = timesteps_sample[i]

        # predict the noise residual
        noise_pred = ema_denoiser_avg(
            torch.cat([latents, labels], dim=1),
            torch.tensor([t] * len(latents)).to(device).long(),
        )
        noise_pred = noise_pred[:, :-2]
        
        prev_t = t-1#max(1, t.item()-1) #max(1, t.item() - (1000//num_inference_steps)) # t-1
        alpha_t = scheduler.alphas_cumprod[t]
        alpha_t_prev = scheduler.alphas_cumprod[prev_t]
        beta_t = 1 - alpha_t
        pred_original_sample = (latents - beta_t ** 0.5 * noise_pred) / alpha_t ** 0.5
        pred_sample_direction = (1 - alpha_t_prev) ** 0.5 * noise_pred
        prev_sample = alpha_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        latents = prev_sample


    return latents 
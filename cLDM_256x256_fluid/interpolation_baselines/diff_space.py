import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset


# LSTM for diffusion latents with size 4*32*32
class TimeConditionedLatentLSTM(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(TimeConditionedLatentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=latent_dim + 1, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout = 0.2)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latent_codes, time_labels):
        # Concatenate time labels to latent codes along the last dimension
        latent_codes_cond = torch.cat((latent_codes, time_labels), dim=-1)  # Shape (batch_size, 20, 4097)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(latent_codes_cond)  # Shape (batch_size, 20, hidden_dim)
        
        # Extract the output for the last step and replicate for 10 steps (future prediction)
        future_out = lstm_out[:, -1:, :]  # Shape (batch_size, 1, hidden_dim)
        #future_out = future_out.repeat(1, 10, 1)  # Repeat for 10 future steps
        future_out = future_out.repeat(1, 1, 1)  # Repeat for 1 future steps

        # Apply final fully connected layer to map LSTM output back to latent space
        output = self.fc(future_out)  # Shape (batch_size, 10, latent_dim)
        
        return output  # Output is of shape (batch_size, 10, 4096)

# Prepare data with 20 input steps and 10 target steps
def prepare_sequences(data, labels, input_len=20, output_len=1):
    X, Y, T = [], [], []
    for i in range(len(data) - input_len - output_len + 1):
        X.append(data[i:i+input_len])
        Y.append(data[i+input_len:i+input_len+output_len])
        T.append(labels[i:i+input_len])  # Time labels for input sequence only
    return torch.stack(X), torch.stack(Y), torch.stack(T)

# Lerp
def lerp(z1, z2, alpha):
    
    z_lerp = torch.stack([ z1 + alph * (z2 - z1) for alph in alpha], dim=0)
    return z_lerp
# Slerp
def slerp(z1, z2, t_array):
    """
    Perform spherical linear interpolation (Slerp) between two latent vectors of shape (1, 4, 32, 32).

    Parameters:
    - z1: torch.Tensor of shape (1, 4, 32, 32), first latent vector.
    - z2: torch.Tensor of shape (1, 4, 32, 32), second latent vector.
    - t_array: torch.Tensor of shape (num_steps,), interpolation values between 0 and 1.

    Returns:
    - interpolated_z: torch.Tensor of shape (num_steps, 1, 4, 32, 32), interpolated latent vectors.
    """

    # Flatten spatial dimensions while keeping batch and channel structure
    batch_size, channels, height, width = z1.shape  # (1, 4, 32, 32)
    z1_flat = z1.view(batch_size, -1)  # Shape: (1, 4*32*32)
    z2_flat = z2.view(batch_size, -1)  # Shape: (1, 4*32*32)

    # Normalize vectors
    z1_norm = z1_flat / torch.norm(z1_flat, dim=1, keepdim=True)
    z2_norm = z2_flat / torch.norm(z2_flat, dim=1, keepdim=True)

    # Compute angle between vectors
    dot_product = torch.sum(z1_norm * z2_norm, dim=1, keepdim=True)  # Shape: (1, 1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Ensure numerical stability
    theta = torch.acos(dot_product)  # Shape: (1, 1)

    # Avoid division by zero for nearly identical vectors
    sin_theta = torch.sin(theta)
    sin_theta = torch.where(sin_theta == 0, torch.tensor(1e-6, device=sin_theta.device), sin_theta)  # Prevent division by zero

    # Compute slerp interpolation for each t in t_array
    interpolated_z = torch.stack([
        (torch.sin((1 - t) * theta) / sin_theta) * z1_flat +
        (torch.sin(t * theta) / sin_theta) * z2_flat
        for t in t_array
    ], dim=0)  # Shape: (num_steps, 1, 4*32*32)

    # Reshape back to original spatial dimensions
    interpolated_z = interpolated_z.view(len(t_array), batch_size, channels, height, width)

    return interpolated_z    

# Taylor's Expansion
def first_derivative(Z: torch.Tensor, dt) -> torch.Tensor:
    """
    Central differences for first derivative. Z: (T, D).
    dt: scalar float OR tensor of shape (T,) giving timestamps.
    Returns: dz (T, D), with one-sided differences at ends.
    """
    assert Z.ndim == 2, "Z must be (T, D)"
    T, D = Z.shape
    dz = torch.empty_like(Z)

    if isinstance(dt, (float, int)) or (isinstance(dt, torch.Tensor) and dt.ndim == 0):
        dt = float(dt)
        dz[1:-1] = (Z[2:] - Z[:-2]) / (2.0 * dt)
        dz[0]    = (Z[1] - Z[0]) / dt
        dz[-1]   = (Z[-1] - Z[-2]) / dt
    else:
        # variable time steps: dt is timestamps of shape (T,)
        t = dt.to(Z.dtype).to(Z.device)
        denom_mid = (t[2:] - t[:-2]).unsqueeze(-1)  # (T-2, 1)
        dz[1:-1] = (Z[2:] - Z[:-2]) / denom_mid
        dz[0]    = (Z[1] - Z[0]) / (t[1] - t[0])
        dz[-1]   = (Z[-1] - Z[-2]) / (t[-1] - t[-2])
    return dz

def second_derivative(Z: torch.Tensor, dt) -> torch.Tensor:
    """
    Central differences for second derivative. Z: (T, D).
    dt: scalar float OR tensor of shape (T,) giving timestamps.
    Returns: ddz (T, D), with one-sided approximations at ends.
    """
    T, D = Z.shape
    ddz = torch.zeros_like(Z)

    if isinstance(dt, (float, int)) or (isinstance(dt, torch.Tensor) and dt.ndim == 0):
        dt = float(dt)
        dt2 = dt * dt
        ddz[1:-1] = (Z[2:] - 2.0 * Z[1:-1] + Z[:-2]) / dt2
        ddz[0]    = (Z[2] - 2.0 * Z[1] + Z[0]) / dt2
        ddz[-1]   = (Z[-1] - 2.0 * Z[-2] + Z[-3]) / dt2
    else:
        # variable time steps: use local forward/backward spacings
        t = dt.to(Z.dtype).to(Z.device)
        h_f = (t[2:] - t[1:-1]).unsqueeze(-1)   # (T-2, 1)
        h_b = (t[1:-1] - t[:-2]).unsqueeze(-1)  # (T-2, 1)
        denom = (h_f + h_b)
        ddz[1:-1] = 2.0 * ((Z[2:] - Z[1:-1]) / (h_f * denom) - (Z[1:-1] - Z[:-2]) / (h_b * denom))
        # crude one-sided at ends
        hf0 = (t[1] - t[0])
        hf1 = (t[2] - t[1])
        ddz[0]  = 2.0 * ((Z[1] - Z[0]) / (hf0 * (hf0 + hf1)) - 0.0)
        hbN1 = (t[-1] - t[-2])
        hbN2 = (t[-2] - t[-3])
        ddz[-1] = 2.0 * (0.0 - (Z[-1] - Z[-2]) / (hbN1 * (hbN1 + hbN2)))
    return ddz

def predict_next_latent_taylor1(z_t: torch.Tensor,
                                dz_t: torch.Tensor,
                                dt) -> torch.Tensor:
    """
    First-order Taylor extrapolation:
        z_{t+dt} ≈ z_t + dz_t * dt
    """
    if isinstance(dt, torch.Tensor):
        dt = float(dt.item())

    return z_t + dz_t * dt

def predict_next_latent_taylor2(z_t: torch.Tensor,
                                dz_t: torch.Tensor,
                                ddz_t: torch.Tensor,
                                dt) -> torch.Tensor:
    """
    Second-order Taylor extrapolation: z_{t+dt} ≈ z_t + dz_t*dt + 0.5*ddz_t*dt^2
    """
    if isinstance(dt, torch.Tensor):
        dt = float(dt.item())
    return z_t + dz_t * dt + 0.5 * ddz_t * (dt * dt)
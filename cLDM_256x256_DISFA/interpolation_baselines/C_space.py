import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset

# LSTM for cebra latents with dim = d
class TimeConditionedLatentLSTM(nn.Module):
    def __init__(self, latent_dim=8, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=latent_dim + 1, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latent_codes, time_labels):
        x = torch.cat((latent_codes, time_labels), dim=-1)  # (batch, seq_len, 4)
        lstm_out, _ = self.lstm(x)
        final_out = lstm_out[:, -1:, :]  # (batch, 1, hidden_dim)
        pred = self.fc(final_out)       # (batch, 1, latent_dim)
        return pred

def prepare_sequences(latent_tensor, time_tensor, input_len=20, output_len=1):
    X, Y, T = [], [], []
    total_steps = latent_tensor.shape[0]
    for i in range(total_steps - input_len - output_len + 1):
        X.append(latent_tensor[i:i+input_len])
        Y.append(latent_tensor[i+input_len:i+input_len+output_len])
        T.append(time_tensor[i:i+input_len])
    return torch.stack(X), torch.stack(Y), torch.stack(T)

# Lerp
def lerp(z1, z2, alpha):
    z_lerp = z1 + alpha * (z2 - z1)
    return z_lerp

# Slerp
def slerp(z1, z2, t_array):
    """
    Perform spherical linear interpolation (Slerp) between two 2D NumPy arrays.

    Parameters:
    - z1: np.ndarray of shape (M, N), first vector.
    - z2: np.ndarray of shape (M, N), second vector.
    - t_array: np.ndarray of shape (num_steps,), interpolation values between 0 and 1.

    Returns:
    - interpolated_z: np.ndarray of shape (num_steps, M, N), interpolated vectors.
    """
    # Normalize vectors with small epsilon for numerical stability
    eps = 1e-8
    z1_norm = z1 / (np.linalg.norm(z1) + eps)
    z2_norm = z2 / (np.linalg.norm(z2) + eps)

    # Compute angle between vectors
    dot_product = np.clip(np.sum(z1_norm * z2_norm), -1.0, 1.0)  # Ensure numerical stability
    theta = np.arccos(dot_product)  # Compute angle

    # Avoid division by zero for nearly identical vectors, fallback to linear interpolation (LERP)
    if np.abs(theta) < 1e-6:
        return np.array([(1 - t) * z1 + t * z2 for t in t_array])

    # Compute SLERP interpolation
    sin_theta = np.sin(theta)
    interpolated_z = np.array([
        (np.sin((1 - t) * theta) / sin_theta) * z1 +
        (np.sin(t * theta) / sin_theta) * z2
        for t in t_array
    ])

    return interpolated_z

# Taylor Expansion
def first_derivative(latent_vectors, delta_t=1.0):
    """Central differences (ends use one-sided). Returns shape (T, D)."""
    Z = np.asarray(latent_vectors)
    T, D = Z.shape
    dz = np.empty_like(Z)
    if np.isscalar(delta_t):
        dt = float(delta_t)
        dz[1:-1] = (Z[2:] - Z[:-2]) / (2*dt)
        dz[0]    = (Z[1] - Z[0]) / dt
        dz[-1]   = (Z[-1] - Z[-2]) / dt
    else:
        # variable time steps: delta_t is array of timestamps length T
        t = np.asarray(delta_t).astype(float)
        dz[1:-1] = (Z[2:] - Z[:-2]) / (t[2:,None] - t[:-2,None])
        dz[0]    = (Z[1]  - Z[0])  / (t[1] - t[0])
        dz[-1]   = (Z[-1] - Z[-2]) / (t[-1] - t[-2])
    return dz

def second_derivative(latent_vectors, delta_t=1.0):
    """Central second differences (ends use one-sided). Returns shape (T, D)."""
    Z = np.asarray(latent_vectors)
    T, D = Z.shape
    ddz = np.zeros_like(Z)
    if np.isscalar(delta_t):
        dt = float(delta_t)
        ddz[1:-1] = (Z[2:] - 2*Z[1:-1] + Z[:-2]) / (dt**2)
        ddz[0]    = (Z[2] - 2*Z[1] + Z[0])       / (dt**2)
        ddz[-1]   = (Z[-1] - 2*Z[-2] + Z[-3])    / (dt**2)
    else:
        # variable time steps: use local dt forward/backward
        t = np.asarray(delta_t).astype(float)
        h_f = t[2:] - t[1:-1]
        h_b = t[1:-1] - t[:-2]
        ddz[1:-1] = 2 * (
            (Z[2:] - Z[1:-1]) / (h_f[:,None] * (h_f[:,None] + h_b[:,None])) -
            (Z[1:-1] - Z[:-2]) / (h_b[:,None] * (h_f[:,None] + h_b[:,None]))
        )
        # crude one-sided at ends
        hf0 = t[1]-t[0]; hf1 = t[2]-t[1]
        ddz[0]  = 2 * ((Z[1]-Z[0])/(hf0*(hf0+hf1)) - 0)
        hbN1 = t[-1]-t[-2]; hbN2 = t[-2]-t[-3]
        ddz[-1] = 2 * (0 - (Z[-1]-Z[-2])/(hbN1*(hbN1+hbN2)))
    return ddz

def predict_next_latent_taylor1(z_t, dz_t, dt):
    """first-order Taylor: z_{t+dt} ≈ z_t + dz_t*dt."""
    return z_t + dz_t * dt
   
def predict_next_latent_taylor2(z_t, dz_t, ddz_t, dt):
    """Second-order Taylor: z_{t+dt} ≈ z_t + dz_t*dt + 0.5*ddz_t*dt^2."""
    return z_t + dz_t * dt + 0.5 * ddz_t * (dt**2)

# Cubic B-spline
from scipy.interpolate import splprep, splev
import numpy as np

def spline_interpolation(Z_data, s=0.0, num_points=None):
    """
    Fit a spline curve through latent trajectory points.

    Parameters
    ----------
    Z_data : np.ndarray
        Input trajectory of shape (T, dim).
    s : float
        Smoothing factor. Use s=0.0 for exact interpolation.
    num_points : int or None
        Number of points to evaluate on the spline.
        If None, uses the original number of time points.

    Returns
    -------
    spline_curve : np.ndarray
        Interpolated spline trajectory of shape (num_points, dim).
    tck : tuple
        Spline representation.
    u_new : np.ndarray
        Parameter values used to evaluate the spline.
    """
    Z_data = np.asarray(Z_data)

    if Z_data.ndim != 2:
        raise ValueError("Z_data must have shape (T, dim)")

    # Fit spline
    tck, u = splprep(Z_data.T, s=s)

    # Evaluate at original or new points
    if num_points is None:
        u_new = u
    else:
        u_new = np.linspace(0, 1, num_points)

    spline_curve = np.array(splev(u_new, tck)).T

    return spline_curve, tck, u_new

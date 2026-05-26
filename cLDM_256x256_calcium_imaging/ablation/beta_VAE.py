# Beta-VAE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.ReLU(),
            nn.Linear(2048, 512), nn.ReLU()
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, 2048), nn.ReLU(),
            nn.Linear(2048, input_dim)
        )

    def forward(self, z):
        return self.net(z)

class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta=0.0):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim)
        self.beta = beta
        self.input_dim = input_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, x, x_recon, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean') * self.input_dim
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + self.beta * kl, recon_loss, kl


def anneal_beta(epoch, warmup=20, max_beta=4.0):
    return min(max_beta, max_beta * epoch / warmup)        
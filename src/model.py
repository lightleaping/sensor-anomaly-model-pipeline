import torch
from torch import nn


class SensorAutoEncoder(nn.Module):
    def __init__(self, input_dim: int = 4, latent_dim: int = 2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


def reconstruction_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> torch.Tensor:
    return torch.mean((original - reconstructed) ** 2, dim=1)

from __future__ import annotations

import torch
from torch import nn


class SensorAutoEncoder(nn.Module):
    """Small dense autoencoder for one row of scaled sensor measurements."""

    def __init__(
        self,
        input_dim: int = 4,
        latent_dim: int = 2,
        hidden_dim: int = 8,
    ) -> None:
        super().__init__()
        if input_dim < 1 or latent_dim < 1 or hidden_dim < 1:
            raise ValueError("Model dimensions must be positive")

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D tensor, got shape {tuple(x.shape)}")
        return self.decoder(self.encoder(x))


def reconstruction_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> torch.Tensor:
    if original.shape != reconstructed.shape:
        raise ValueError(
            "Original and reconstructed tensors must have identical shapes"
        )
    return torch.mean((original - reconstructed) ** 2, dim=1)

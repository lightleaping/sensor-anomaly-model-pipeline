import pytest
import torch

from src.model import SensorAutoEncoder, reconstruction_error


def test_autoencoder_output_and_error_shape() -> None:
    model = SensorAutoEncoder(input_dim=4, latent_dim=2, hidden_dim=8)
    samples = torch.randn(5, 4)
    reconstructed = model(samples)
    errors = reconstruction_error(samples, reconstructed)

    assert reconstructed.shape == samples.shape
    assert errors.shape == (5,)
    assert torch.all(errors >= 0)


def test_autoencoder_rejects_non_matrix_input() -> None:
    model = SensorAutoEncoder()
    with pytest.raises(ValueError, match="2D tensor"):
        model(torch.randn(4))


def test_reconstruction_error_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        reconstruction_error(torch.randn(2, 4), torch.randn(2, 3))

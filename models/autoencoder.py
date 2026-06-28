import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, input_length: int = 512, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1: input_length -> input_length/2
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            # Block 2: /2 -> /4
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Block 3: /4 -> /8
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Block 4: /8 -> /16
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # Compute flattened size after convolutions
        self._flat_size = 128 * (input_length // 16)
        self.fc = nn.Linear(self._flat_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ConvDecoder(nn.Module):
    def __init__(self, input_length: int = 512, latent_dim: int = 32):
        super().__init__()
        self._seq_len = input_length // 16
        self.fc = nn.Linear(latent_dim, 128 * self._seq_len)

        self.decoder = nn.Sequential(
            # Block 1: /16 -> /8
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Block 2: /8 -> /4
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Block 3: /4 -> /2
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            # Block 4: /2 -> full length
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 128, self._seq_len)
        return self.decoder(x)


class PumpAutoencoder(nn.Module):
    """
    Convolutional autoencoder for pump vibration anomaly detection.

    Trained exclusively on normal operation data. At inference time,
    high reconstruction error (MSE) signals an anomaly.

    Input shape:  (batch, 1, input_length)   -- single-channel vibration window
    Output shape: (batch, 1, input_length)   -- reconstructed signal
    """

    def __init__(self, input_length: int = 512, latent_dim: int = 32):
        super().__init__()
        assert input_length % 16 == 0, "input_length must be divisible by 16"
        self.encoder = ConvEncoder(input_length, latent_dim)
        self.decoder = ConvDecoder(input_length, latent_dim)
        self.input_length = input_length
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample MSE between input and reconstruction."""
        x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=(1, 2))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(self, Encoder).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            nn.LeakyReLU(0.01),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            nn.LeakyReLU(0.01),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            nn.Flatten(),
            nn.Linear(3136, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x

import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-1, :-1]


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=3136),
            Reshape(-1, 64, 7, 7),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0
            ),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            Trim(),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        return x

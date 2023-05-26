import torch
import torch.nn as nn

from app.autoencoder.encoder import Encoder
from app.autoencoder.decoder import Decoder


class DenoisingAutoencoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device) -> None:
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = encoder
        self.encoder.to(device)

        self.decoder = decoder
        self.decoder.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

"""
model.py

Defines the OceanEncoderDecoder PyTorch model class for subsurface ocean temperature prediction.

The model uses an encoder-decoder architecture:
1. Encoder: Compresses the 10-dimensional input feature space (representing geographic, 
   seasonal, and surface satellite observations) down to a compact 24-dimensional latent embedding.
2. Latent Embedding: Represents a simplified representation of the surface state.
3. Decoder: Expands the latent embedding back out to reconstruct the vertical temperature profile 
   across the 8 target ocean depth levels (0m to 500m).
"""

import torch
import torch.nn as nn


class OceanEncoderDecoder(nn.Module):
    def __init__(self, input_dim=10, embedding_dim=24, output_dim=8):
        super(OceanEncoderDecoder, self).__init__()
        
        # Encoder Network: Compresses input state to a latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
            nn.ReLU()
        )
        
        # Decoder Network: Reconstructs subsurface temperatures from the embedding
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def encode(self, x):
        """
        Runs the inputs through the encoder network to produce the latent embeddings.
        This is useful for extracting/visualizing the low-dimensional representations.
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Runs the latent embeddings through the decoder network to predict the target values.
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Performs a full forward pass through both the encoder and decoder.
        """
        z = self.encode(x)
        y_pred = self.decode(z)
        return y_pred

import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, n_clusters, intermediate_sizes):
        super().__init__()
        self.embedding_size = embedding_size
        self.alpha = 10
        self.a_enc_centers = nn.Parameter(
            torch.zeros(n_clusters, embedding_size, requires_grad=True, dtype=torch.float), requires_grad=True)
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.intermediate_sizes = intermediate_sizes

        encoder_dim = self.intermediate_sizes + [self.embedding_size]
        decoder_dim = self.intermediate_sizes[::-1] + [self.input_size]
        encoder_activations = [nn.LeakyReLU for _ in range(len(self.intermediate_sizes))] + [None]
        decoder_activations = [nn.LeakyReLU for _ in range(len(self.intermediate_sizes))] + [None]

        self._encoder = nn.Sequential(*self._spec2seq(self.input_size, encoder_dim, encoder_activations))
        self._decoder = nn.Sequential(*self._spec2seq(self.embedding_size, decoder_dim, decoder_activations))

    def forward(self, x):
        ae_embedding = self._encoder(x)
        reconstructed = self._decoder(ae_embedding)
        return ae_embedding, reconstructed

    def _spec2seq(self, input, dimentions, activations):
        layers = []
        for dim, act in zip(dimentions, activations):
            layer = nn.Linear(input, dim)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if act:
                layers.append(act())
            input = dim
        return layers

import torch
from torch import nn
from .ListModule import ListModule
from torch.nn import functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, latent_dim):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Linear(input_dim, encoding_dim)
        self.latent_layer = nn.Linear(self.encoding_dim, self.latent_dim)
        self.rec_latent_layer = nn.Linear(self.latent_dim, self.encoding_dim)
        self.decoder = nn.Linear(self.encoding_dim, self.input_dim)
        self.act = nn.Sigmoid()

    def forward(self, X):
        enc = self.act(self.encoder(X))
        z = self.act(self.latent_layer(enc))
        # rec = self.act(F.linear(z, self.latent_layer.weight.t()))
        # out = self.act(F.linear(rec, self.encoder.weight.t()))

        rec = self.act(self.rec_latent_layer(z))
        out = self.act(self.decoder(rec))
        return z, out


class MDA(nn.Module):
    def __init__(self, input_dims, encoding_dims, latent_dim):
        super(MDA, self).__init__()

        self.input_dims = input_dims
        self.encoding_dims = encoding_dims
        self.abstract_layer_size = sum(encoding_dims)
        self.latent_dim = latent_dim
        self.encoders = [nn.Linear(input_dims[i], encoding_dims[i]) for i in range(len(input_dims))]
        self.init_weight(self.encoders)
        self.encoders = ListModule(*self.encoders)
        self.latent_layer = nn.Linear(self.abstract_layer_size, self.latent_dim)
        self.rec_latent_layer = nn.Linear(self.latent_dim, self.abstract_layer_size)
        self.rec_encoding_layers = [
            nn.Linear(self.abstract_layer_size, encoding_dims[i]) for i in range(len(input_dims))]

        self.init_weight(self.rec_encoding_layers)
        self.rec_encoding_layers = ListModule(*self.rec_encoding_layers)
        self.decoders = [nn.Linear(self.encoding_dims[i], input_dims[i])
                         for i in range(len(input_dims))]
        self.init_weight(self.decoders)
        self.decoders = ListModule(*self.decoders)
        self.act = nn.Sigmoid()

        self.init_weight(self.latent_layer)
        self.init_weight(self.rec_latent_layer)

    def forward(self, X):
        enc = torch.cat([self.act(self.encoders[i](X[i]))
                         for i in range(len(self.input_dims))], dim=1)
        # print(enc.shape)
        z = self.act(self.latent_layer(enc))
        # print(z.shape)
        rec = self.act(self.rec_latent_layer(z))
        # print(rec.shape)
        # rec = torch.chunk(rec, len(X), dim=1)
        rec_enc = [self.act(self.rec_encoding_layers[i](rec)) for i in range(len(self.input_dims))]
        out = [self.decoders[i](rec_enc[i]) for i in range(len(self.input_dims))]

        return z, out

    def init_weight(self, layer):
        if isinstance(layer, list):
            for i in range(len(layer)):
                nn.init.xavier_uniform_(layer[i].weight)
        else:
            nn.init.xavier_uniform_(layer.weight)

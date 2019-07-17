import torch
from torch import nn
from .ListModule import ListModule
from torch.nn import functional as F


def kl_divergence(p, q):
    '''
    args:
        2 tensors `p` and `q`
    returns:
        kl divergence between the softmax of `p` and `q`
    '''
    p = F.softmax(p)
    q = F.softmax(q)

    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


class Attention(nn.Module):

    def __init__(self, att_size):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.att_w = nn.Linear(att_size, 1, bias=False)
        nn.init.xavier_uniform_(self.att_w.weight)

    def forward(self, input):
        att = self.att_w(input.permute(0, 2, 1)).squeeze(-1)
        out = self.softmax(att)
        return out.unsqueeze(2)


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


class MDA_Layer(nn.Module):
    def __init__(self, input_dims, encoding_dims):
        super(MDA_Layer, self).__init__()

        self.input_dims = input_dims
        self.encoding_dims = encoding_dims
        self.abstract_layer_size = encoding_dims[0]
        self.encoders = [nn.Linear(input_dims[i], encoding_dims[i]) for i in range(len(input_dims))]
        self.init_weight(self.encoders)
        self.encoders = ListModule(*self.encoders)
        self.attention = Attention(encoding_dims[0])
        self.act = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, X, attend=False):
        enc = torch.stack([self.dropout(self.act(self.encoders[i](X[i])))
                           for i in range(len(self.input_dims))], dim=2)
        att = self.attention(enc)
        enc = att * enc.permute(0, 2, 1)
        if attend:
            return enc.sum(1), [self.encoders[i].weight.t() for i in range(len(X))]
        else:
            return [enc[:, i, :] for i in range(len(X))], [self.encoders[i].weight.t() for i in range(len(X))]

    def init_weight(self, layer):
        if isinstance(layer, list):
            for i in range(len(layer)):
                nn.init.xavier_uniform_(layer[i].weight)
        else:
            nn.init.xavier_uniform_(layer.weight)


class MDA(nn.Module):
    def __init__(self, input_dims, encoding_dims, latent_dim):
        super(MDA, self).__init__()

        self.input_dims = input_dims
        self.encoding_dims = encoding_dims
        self.abstract_layer_size = encoding_dims[0]
        self.latent_dim = latent_dim
        self.encoders = MDA_Layer(input_dims, encoding_dims)
        self.latent_layer = nn.Linear(self.abstract_layer_size, self.latent_dim)
        self.attention = Attention(encoding_dims[0])
        self.act = nn.Sigmoid()
        # self.bn = nn.BatchNorm1d(encoding_dims[0])
        self.dropout = nn.Dropout(0.1)
        self.init_weight(self.latent_layer)

    def forward(self, X):
        enc, enc_weights = self.encoders(X, attend=True)
        z = self.dropout(self.act(self.latent_layer(enc)))
        latent_rec = self.dropout(self.act(F.linear(z, self.latent_layer.weight.t())))

        out = [self.act(F.linear(latent_rec, enc_weights[i])) for i in range(len(self.input_dims))]
        return z, enc, latent_rec, out

    def init_weight(self, layer):
        if isinstance(layer, list):
            for i in range(len(layer)):
                nn.init.xavier_uniform_(layer[i].weight)
        else:
            nn.init.xavier_uniform_(layer.weight)

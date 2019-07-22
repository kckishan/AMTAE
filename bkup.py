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
        self.att_w = nn.Conv1d(att_size, 1, 1)
        nn.init.xavier_uniform_(self.att_w.weight)

    def forward(self, input):
        att = self.att_w(input.permute(0, 2, 1)).squeeze(-1)
        out = self.softmax(att)
        return out.permute(0, 2, 1)


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
        self.lin = nn.Linear(encoding_dims[0], encoding_dims[0])

    def forward(self, X, attend=False):

        enc = torch.stack([self.dropout(self.act(self.encoders[i](X[i])))
                           for i in range(len(self.input_dims))], dim=2)
        enc = enc.permute(0, 2, 1)
        emb_enc = torch.tanh(self.lin(enc))
        attn = self.attention(emb_enc)
        enc = attn * enc
        if attend:
            return enc.sum(1), [self.encoders[i].weight.t() for i in range(len(X))], attn
        else:
            return [enc[:, i, :] for i in range(len(X))], [self.encoders[i].weight.t() for i in range(len(X))], attn

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
        self.rec_latent_layer = nn.Linear(self.latent_dim, self.abstract_layer_size)
        self.rec_decoders = [nn.Linear(self.abstract_layer_size, encoding_dims[i])
                             for i in range(len(input_dims))]
        self.init_weight(self.rec_decoders)
        self.rec_decoders = ListModule(*self.rec_decoders)

        self.decoders = [nn.Linear(encoding_dims[i], input_dims[i]) for i in range(len(input_dims))]
        self.init_weight(self.decoders)
        self.decoders = ListModule(*self.decoders)
        self.attention = Attention(encoding_dims[0])
        self.act = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
        self.init_weight(self.latent_layer)

    def forward(self, X):
        enc, enc_weights, attn = self.encoders(X, attend=True)
        z = self.dropout(self.act(self.latent_layer(enc)))
        latent_rec = self.dropout(self.act(self.rec_latent_layer(z)))
        enc_rec = [self.act(self.rec_decoders[i](latent_rec)) for i in range(len(self.input_dims))]
        out = [self.act(self.decoders[i](enc_rec[i])) for i in range(len(self.input_dims))]
        return z, enc, latent_rec, out, attn.squeeze()

    def init_weight(self, layer):
        if isinstance(layer, list):
            for i in range(len(layer)):
                nn.init.xavier_uniform_(layer[i].weight)
        else:
            nn.init.xavier_uniform_(layer.weight)

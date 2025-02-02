# stolen from https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py

import math
import numpy as np
import torch

EPS = 1e-6


# def setup_pos_embed_1d(min_val=-5, max_val=5, bins=1001, emb_dim=1536):
#     pos_embeddings = positionalencoding1d(emb_dim, bins)
#     positions = np.linspace(min_val, max_val, bins)


class PositionalEncoding1D(torch.nn.Module):
    def __init__(
        self,
        min_val=-5,
        max_val=5,
        bins=1001,
        emb_dim=1536,
        eps=EPS,
        add_learnable_constant=False,
        normalized=False,
    ):
        super().__init__()
        self.eps = eps
        self.min_val = min_val
        self.max_val = max_val
        self.register_buffer("pos_embeddings", positionalencoding1d(emb_dim, bins, normalized=normalized))

        if add_learnable_constant:
            self.learnable_constant = torch.nn.Parameter(0.02 * torch.randn(emb_dim))
        else:
            self.register_buffer("learnable_constant", torch.zeros(emb_dim))

        self.positions = np.linspace(min_val, max_val, bins)

    def forward(self, x):
        x = np.clip(x.cpu().numpy(), self.min_val, self.max_val - self.eps)
        idx = np.digitize(x, self.positions)
        return self.pos_embeddings[idx] + self.learnable_constant


class PositionalEncoding2D(torch.nn.Module):
    def __init__(
        self,
        min_val_1=-5,
        max_val_1=5,
        bins_1=1001,
        min_val_2=-5,
        max_val_2=5,
        bins_2=1001,
        emb_dim=1536,
        add_learnable_constant=False,
        eps=EPS,
        normalized=False,
    ):
        super().__init__()
        self.eps = eps
        self.min_val_1 = min_val_1
        self.max_val_1 = max_val_1
        self.min_val_2 = min_val_2
        self.max_val_2 = max_val_2
        self.register_buffer(
            "pos_embeddings", positionalencoding2d(emb_dim, bins_1, bins_2, normalized=normalized)
        )
        if add_learnable_constant:
            self.learnable_constant = torch.nn.Parameter(0.02 * torch.randn(emb_dim))
        else:
            self.register_buffer("learnable_constant", torch.zeros(emb_dim))

        self.positions_1 = np.linspace(min_val_1, max_val_1, bins_1)
        self.positions_2 = np.linspace(min_val_2, max_val_2, bins_2)

    def forward(self, x1, x2):
        x1 = np.clip(x1.cpu().numpy(), self.min_val_1, self.max_val_1 - self.eps)
        x2 = np.clip(x2.cpu().numpy(), self.min_val_2, self.max_val_2 - self.eps)

        idx_1 = np.digitize(x1, self.positions_1)
        idx_2 = np.digitize(x2, self.positions_2)
        return self.pos_embeddings[:, idx_1, idx_2].T + self.learnable_constant


def positionalencoding1d(d_model, length, normalized=False):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0., length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0., d_model, 2)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    if normalized:
        pe = pe / torch.norm(pe, dim=-1, keepdim=True)
    return pe


def positionalencoding2d(d_model, height, width, normalized=False):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )

    if normalized:
        pe = pe / torch.norm(pe, dim=0, keepdim=True)

    return pe

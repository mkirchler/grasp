import os
import pickle
from functools import partial
from pathlib import Path

import h5py

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from timm.models.layers import DropPath, trunc_normal_
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tte.utils.positional_embeddings import PositionalEncoding1D, PositionalEncoding2D

NAN_VALUE = -9.0


def get_simple_transformer(
    input_dim=1536,
    num_heads=8,
    embed_dim=64,
    depth=4,
    output_size=1,
    age_embed_config=None,
    val_embed_config=None,
    positional_constant=False,
    learnable_positional_factor=False,
    positional_embeddings_normalized=False,
):
    if age_embed_config is None:
        age_embed_config = {
            "use_age_embeddings": False,
            "min_val": 30,
            "max_val": 80,
            "bins": 101,
        }
    if val_embed_config is None:
        val_embed_config = {
            "use_val_embeddings": False,
            "min_val": -5,
            "max_val": 5,
            "bins": 101,
        }
    attn_target = partial(MultiHeadAttention, num_heads=num_heads)
    T = SimpleTransformer(
        input_dim=input_dim,
        embed_dim=embed_dim,
        depth=depth,
        attn_target=attn_target,
        output_size=output_size,
        min_age=age_embed_config["min_val"],
        max_age=age_embed_config["max_val"],
        age_bins=age_embed_config["bins"],
        use_age_embeddings=age_embed_config["use_age_embeddings"],
        min_val=val_embed_config["min_val"],
        max_val=val_embed_config["max_val"],
        val_bins=val_embed_config["bins"],
        use_val_embeddings=val_embed_config["use_val_embeddings"],
        positional_constant=positional_constant,
        learnable_positional_factor=learnable_positional_factor,
        positional_embeddings_normalized=positional_embeddings_normalized,
    )
    return T


## TODO: re-check init
## TODO: norm layer more often?
class SimpleTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        depth,
        classifier_feature="cls_token",
        drop_path_type="progressive",
        drop_path_rate=0.1,
        drop_rate=0.0,
        mlp_ratio=4,
        attn_target=None,
        layer_scale_type=None,
        layer_scale_init_value=1e-4,
        layer_norm_eps=1e-6,
        output_size=1,
        min_age=30,
        max_age=80,
        age_bins=101,
        use_age_embeddings=True,
        min_val=-5,
        max_val=5,
        val_bins=101,
        use_val_embeddings=True,
        positional_constant=False,
        positional_embeddings_normalized=False,
        learnable_positional_factor=False,
    ):
        super().__init__()
        if isinstance(output_size, (list, tuple)):
            assert len(output_size) == 2
            output_size, aux_output_size = output_size
            self.aux_head = nn.Linear(embed_dim, aux_output_size)
        else:
            self.aux_head = None
        self.head = nn.Linear(embed_dim, output_size)

        norm_layer = partial(nn.LayerNorm, eps=layer_norm_eps)
        self.norm = norm_layer(embed_dim)

        self.use_age_embeddings = use_age_embeddings
        if use_age_embeddings:
            self.age_only_embeddings = PositionalEncoding1D(
                min_val=min_age,
                max_val=max_age,
                bins=age_bins,
                emb_dim=embed_dim,
                add_learnable_constant=positional_constant,
                normalized=positional_embeddings_normalized,
            )

        self.use_val_embeddings = use_val_embeddings
        if use_val_embeddings:
            self.val_only_embeddings = PositionalEncoding1D(
                min_val=min_val,
                max_val=max_age,
                bins=age_bins,
                emb_dim=embed_dim,
                add_learnable_constant=positional_constant,
                normalized=positional_embeddings_normalized,
            )

        if use_age_embeddings and use_val_embeddings:
            self.both_embeddings = PositionalEncoding2D(
                min_val_1=min_age,
                max_val_1=max_age,
                bins_1=age_bins,
                min_val_2=min_val,
                max_val_2=max_val,
                bins_2=val_bins,
                emb_dim=embed_dim,
                normalized=positional_embeddings_normalized,
            )

        # gets exp'd to stay positive
        if learnable_positional_factor:
            self.positional_factor = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("positional_factor", torch.zeros(1))

        assert classifier_feature in ["cls_token", "global_pool"]
        self.classifier_feature = classifier_feature

        self.token_embed = nn.Linear(input_dim, embed_dim)
        if classifier_feature == "cls_token":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=0.02)

        else:
            self.cls_token = None
        # stochastic depth decay rule
        assert drop_path_type in [
            "progressive",
            "uniform",
        ], f"Drop path types are: [progressive, uniform]. Got {drop_path_type}."
        if drop_path_type == "progressive":
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(depth)
            ]
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.apply(self._init_weights)

    def prepare_tokens(self, x, ages, values):
        x = self.token_embed(x)

        has_age = ages != NAN_VALUE
        has_val = values != NAN_VALUE

        has_both = has_age & has_val
        has_age_only = has_age & ~has_val
        has_val_only = ~has_age & has_val

        if self.use_age_embeddings and has_age_only.any():
            # print("using age only embeddings")
            x[has_age_only] = x[
                has_age_only
            ] + self.positional_factor.exp() * self.age_only_embeddings(
                ages[has_age_only]
            )
        if self.use_val_embeddings and has_val_only.any():
            # print("using val only embeddings")
            x[has_val_only] = x[
                has_val_only
            ] + self.positional_factor.exp() * self.val_only_embeddings(
                values[has_val_only]
            )
        if self.use_age_embeddings and self.use_val_embeddings and has_both.any():
            # print("using both embeddings -- both")
            x[has_both] = x[
                has_both
            ] + self.positional_factor.exp() * self.both_embeddings(
                ages[has_both], values[has_both]
            )
        if self.use_age_embeddings and not self.use_val_embeddings and has_both.any():
            # print("using both embeddings -- age only")
            x[has_both] = x[
                has_both
            ] + self.positional_factor.exp() * self.age_only_embeddings(ages[has_both])
        if not self.use_age_embeddings and self.use_val_embeddings and has_both.any():
            # print("using both embeddings -- val only")
            # print(x.dtype, values.dtype, self.val_only_embeddings.pos_embeddings.dtype, self.positional_factor.dtype)

            x[has_both] = x[
                has_both
            ] + self.positional_factor.exp() * self.val_only_embeddings(
                values[has_both]
            )

        x = self.pos_drop(x)
        return x

    def forward_features(self, x, ages, values):
        x = self.prepare_tokens(x, ages, values)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        for block in self.blocks:
            x = block(x)

        if self.classifier_feature == "cls_token":
            x = x[:, 0]
        elif self.classifier_feature == "global_pool":
            x = x[:, 0, ...].mean(dim=1)
        x = self.norm(x)
        return x

    def forward(self, x):
        ages = x["ages"]
        values = x["values"]
        x = x["data"]

        x = self.forward_features(x, ages=ages, values=values)
        if self.aux_head is not None:
            return self.head(x), self.aux_head(x)
        else:
            x = self.head(x)
            return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        attn_target,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale_type=None,  # from cait; possible values are None, "per_channel", "scalar"
        layer_scale_init_value=1e-4,  # from cait; float
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if isinstance(attn_target, nn.Module):
            self.attn = attn_target
        else:
            self.attn = attn_target(dim=dim)

        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale_type = layer_scale_type

        # Layerscale
        if self.layer_scale_type is not None:
            assert self.layer_scale_type in [
                "per_channel",
                "scalar",
            ], f"Found Layer scale type {self.layer_scale_type}"
            if self.layer_scale_type == "per_channel":
                # one gamma value per channel
                gamma_shape = [1, 1, dim]
            elif self.layer_scale_type == "scalar":
                # single gamma value for all channels
                gamma_shape = [1, 1, 1]
            # two gammas: for each part of the fwd in the encoder
            self.layer_scale_gamma1 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )
            self.layer_scale_gamma2 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )

    def forward(self, x):
        if self.layer_scale_type is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)) * self.layer_scale_gamma1)
            x = x + self.drop_path(self.mlp(self.norm2(x)) * self.layer_scale_gamma2)
        return x

    def extra_repr(self) -> str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ""
        for p in self.named_parameters():
            name = p[0].split(".")[0]
            if name not in named_modules:
                string_repr = (
                    string_repr
                    + "("
                    + name
                    + "): "
                    + "tensor("
                    + str(tuple(p[1].shape))
                    + ", requires_grad="
                    + str(p[1].requires_grad)
                    + ")\n"
                )

        return string_repr


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        # problem: only for new torch versions?
        use_opt_attention=True,
    ):
        super().__init__()
        self.use_opt_attention = use_opt_attention
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_dropout_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def manual_attention(self, query, key, value):
        scale = 1 / query.shape[-1] ** 0.5
        query = query * scale
        attn = query @ key.transpose(-2, -1)
        attn = attn.softmax(-1)
        attn = F.dropout(attn, self.attn_dropout_p)
        return attn @ value

    def fast_attention(self, query, key, value):
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.attn_dropout_p
        )

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        if not self.use_opt_attention:
            x = self.manual_attention(q, k, v)
        else:
            x = self.fast_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

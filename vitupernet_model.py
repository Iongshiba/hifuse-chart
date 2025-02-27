import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

from hifuse_model import PatchEmbed as PatchEmbeddings
from config import *


class PatchEmbed(nn.Module):
    # REUSE FROM HIFUSE
    pass


class main_model(nn.Module):

    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        embed_dim=1024,
        num_trans=(4, 6, 8, 6),
        num_heads=(16, 16, 16, 16),
        norm_layer=nn.LayerNorm,
        **kwargs
    ):
        super().__init__()


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the position embeddings.
    """

    def __init__(
        self,
        patch_size,
        patch_num,
        in_chans,
        embed_dim,
        norm_layer=None,
        dropout_prob=0.0,
    ):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(
            patch_size=patch_size,
            in_c=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

        self.position_embeddings = nn.parameter.Parameter(
            torch.rand(1, patch_num, embed_dim)
        )

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x, _, _ = self.patch_embeddings(x)
        x = x + self.position_embeddings
        x = self.dropout(x)

        # [B, patch, embed] -> [B, patch + 1, embed]
        return x


class Attention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        # square root of dk
        self.scale = head_dim**-0.5
        # [B, patch + 1, embed] -> [B, patch + 1, head_dim * 3]
        self.to_qkv = nn.Linear(embed_dim, head_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, P_, C = x.shape

        # [B, patch + 1, head_dim * 3] -> [3, B, patch + 1, head_dim]
        qkv = self.to_qkv(x).reshape(3, B_, P_, C // 3)

        # .. -> [B, patch + 1, head_dim]
        q, k, v = qkv.unbind(0)

        # Attention calculation
        attn = self.softmax(q @ k.transpose(-2, -1) * self.scale) @ v

        return attn


class MLP(nn.Module):
    pass


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

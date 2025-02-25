import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

from hifuse_model import PatchEmbed


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

        ###### Patch Embeddings #######

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_c=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if norm_layer else None,
        )


class PatchEmbed(nn.Module):
    # REUSE FROM HIFUSE
    pass


class PositionalEncoding(nn.Module):
    pass


class Attention(nn.Module):
    pass


class MLP(nn.Module):
    pass

import torch

from torch import nn
from torchvision.ops.misc import MLP

import torch.nn.functional as F


class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, hidden_dim):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # 4 is the number of coords in a bounding box
        self.bbox_embed = MLP(
            in_channels=hidden_dim,
            hidden_channels=[hidden_dim, hidden_dim, 4],
            activation_layer=nn.ReLU,
        )
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, 1)

        ###### Transformer Setting ######

    def forward(self, x):
        pass


class TransformerEncoder(nn.Module):
    def __init__(self, out_channels, ffn_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.attn = nn.MultiheadAttention(out_channels, num_heads, dropout)
        self.norm1 = nn.LayerNorm(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = MLP(
            out_channels,
            [ffn_dim, out_channels],
            dropout=dropout,
            activation_layer=nn.ReLU,
        )
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # self-attention
        x_attn = self.attn(x, x, x)
        x = x + self.dropout1(x_attn)
        x = self.norm1(x)

        x_ffn = self.ffn(x)
        x = x + self.dropout2(x_ffn)
        x = self.norm2(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, out_channels, ffn_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.self_attn = nn.MultiheadAttention(out_channels, num_heads, dropout)
        self.norm1 = nn.LayerNorm(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(out_channels, num_heads, dropout)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = MLP(
            out_channels,
            [ffn_dim, out_channels],
            dropout=dropout,
            activation_layer=nn.ReLU,
        )
        self.norm3 = nn.LayerNorm(out_channels)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, emb):
        # self-attention
        x_attn = self.self_attn(x, x, x)
        x = x + self.dropout1(x_attn)
        x = self.norm1(x)

        # cross-attention
        cross_attn = self.cross_attn(x, emb, emb)
        x = x + self.dropout2(cross_attn)
        x = self.norm2(x)

        x_ffn = self.ffn(x)
        x = x + self.dropout3(x_ffn)
        x = self.norm3(x)

        return x

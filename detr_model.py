import copy
import torch

from torch import nn
from torchvision.ops.misc import MLP

import torch.nn.functional as F


class DETR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_queries: int,
        hidden_dim: int,
        feedforward_dim: int = 2048,
        num_heads: int = 8,
        encoder_num: int = 6,
        decoder_num: int = 6,
        temperature: int = 10000,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # 4 is the number of coords in a bounding box
        self.bbox_embed = MLP(
            in_channels=hidden_dim,
            hidden_channels=[hidden_dim, hidden_dim, 4],
            activation_layer=nn.ReLU,
        )
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        ###### Transformer Setting ######

        self.encoder_layer = TransformerEncoderLayer(
            hidden_dim, feedforward_dim, num_heads
        )
        self.decoder_layer = TransformerDecoderLayer(
            hidden_dim, feedforward_dim, num_heads
        )

        self.encoder = TransformerEncoder(self.encoder_layer, encoder_num)
        self.decoder = TransformerDecoder(
            self.encoder_layer, decoder_num, nn.LayerNorm(hidden_dim)
        )

        ###### Loss Setting ######

    def _pos_embed(self, x):
        # x = (batch_size, sequence_length, hidden_dim)
        _, H, W, C = x.shape
        embed_dim = self.hidden_dim // 2

        # dim_t = [t^(2*0/hidden_dim), t^(2*1/hidden_dim), t^(2*2/hidden_dim), t^(2*3/hidden_dim)]
        dim_t = torch.arange(0, embed_dim, dtype=torch.float32, device=x.device)
        dim_t = torch.pow(
            self.temperature,
            2 * (dim_t // 2) / embed_dim,
        )

        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=x.device)
        x_pos, y_pos = torch.meshgrid(y_embed, x_embed)

        x_pos = x_pos[:, :, None] / dim_t
        y_pos = y_pos[:, :, None] / dim_t
        x_pos = torch.stack(
            (x_pos[:, :, 0::2].sin(), x_pos[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        y_pos = torch.stack(
            (y_pos[:, :, 0::2].sin(), y_pos[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        pos = torch.cat((x_pos, y_pos), dim=2).unsqueeze(0).permute(0, 3, 1, 2)

        return pos

    def forward(self, x):
        pos_embed = self._pos_embed(x)

        memory = self.encoder(x, pos_embed)
        target = torch.zeros_like(self.query_embed)
        output = self.decoder(target, memory, pos_embed, self.query_embed)

        return output


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, x, pos_embed):
        output = x

        for layer in self.layers:
            output = layer(output, pos_embed)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm

    def forward(self, x, memory, pos_embed, query_embed):
        output = x

        for layer in self.layers:
            output = layer(output, memory, pos_embed, query_embed)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, out_channels, ffn_dim, num_heads, dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
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

    def forward(self, x, pos_embed):
        # positional memoryedding at each encoding layer
        q = k = x + pos_embed

        # self-attention
        x_attn = self.attn(q, k, x)
        x = x + self.dropout1(x_attn)
        x = self.norm1(x)

        # feedforward
        x_ffn = self.ffn(x)
        x = x + self.dropout2(x_ffn)
        x = self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, out_channels, ffn_dim, num_heads, dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
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

    def forward(self, x, memory, pos_embed, query_embed):
        # positional embedding at each decoding layer
        q_self = k_self = x + query_embed

        # self-attention
        self_attn = self.self_attn(q_self, k_self, x)
        x = x + self.dropout1(self_attn)
        x = self.norm1(x)

        # cross-attention with box query and positional embedding
        q_cross = x + query_embed
        k_cross = memory + pos_embed
        cross_attn = self.cross_attn(q_cross, k_cross, memory)
        x = x + self.dropout2(cross_attn)
        x = self.norm2(x)

        x_ffn = self.ffn(x)
        x = x + self.dropout3(x_ffn)
        x = self.norm3(x)

        return x


def _get_clones(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


if __name__ == "__main__":
    x = torch.ones((1, 7, 7, 3))
    detr = DETR(
        1,
        100,
        8,
    )

    print(detr._pos_embed(x))

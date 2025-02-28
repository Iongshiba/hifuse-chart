import einops
from torch import nn, rand, cat

# import numpy as np
from einops.layers.torch import Rearrange
from torchvision.ops.misc import MLP


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels=3,
        patch_size=8,
        emb_size=128,
    ):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_size,
                p2=self.patch_size,
            ),
            nn.Linear(in_channels * self.patch_size * self.patch_size, emb_size),
        )

    def forward(self, x):
        return self.projection(x)


class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.att = nn.MultiheadAttention(
            embed_dim=emb_size, num_heads=num_heads, dropout=dropout
        )

        self.q = nn.Linear(emb_size, emb_size)
        self.k = nn.Linear(emb_size, emb_size)
        self.v = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        return self.att(q, k, v)


class MLP(MLP):
    def __init__(self, in_dim, mlp_dim, dropout):
        super().__init__(
            in_dim, [mlp_dim, in_dim], dropout=dropout, activation_layer=nn.GELU
        )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        return x + res


class ViT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_dim=1,
        img_size=224,
        patch_size=8,
        emb_size=128,
        num_layers=12,
        num_heads=2,
        dropout=0.1,
    ):
        self.in_channels = in_channels
        self.width = img_size
        self.height = img_size
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.patch_embedding = PatchEmbedding(
            in_channels=self.in_channels,
            patch_size=self.patch_size,
            emb_size=self.emb_size,
        )

        self.num_patch = (self.width * self.height) // (self.patch_size**2)
        self.pos_embedding = nn.Parameter(rand(1, self.num_patch + 1, emb_size))
        self.cls_token = nn.Parameter(rand(1, 1, emb_size))

        self.layers = nn.ModuleList([])
        for _ in range(self.num_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(
                    PreNorm(
                        emb_size,
                        Attention(
                            emb_size=emb_size, num_heads=num_heads, dropout=self.dropout
                        ),
                    )
                ),
                ResidualAdd(
                    PreNorm(
                        emb_size,
                        MLP(in_dim=emb_size, mlp_dim=emb_size, dropout=self.dropout),
                    )
                ),
            )
            self.layers.append(transformer_block)

        self.head = nn.Sequential(nn.LayerNorm(emb_size), nn.Linear(emb_size, out_dim))

    def forward(self, img):
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        cls_token = einops.repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = cat([x, cls_token], dim=1)
        x += self.pos_embedding[:, : (n + 1)]

        for i in range(self.num_layer):
            x = self.layers[i](x)

        return self.head(x[:, 0, :])

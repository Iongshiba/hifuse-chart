from torch import nn, cat, rand, _assert
from torchvision.ops.misc import MLP
from einops.layers.torch import Rearrange
from einops import repeat


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channel: int,
        emb_size: int,
        patch_size: int,
    ):
        super().__init__()
        self.project = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.Linear((in_channel * patch_size * patch_size), emb_size),
        )

    def forward(self, x):
        return self.project(x)


class Embedding(nn.Module):
    def __init__(
        self,
        in_channel: int,
        img_size: int,
        emb_size: int,
        patch_size: int,
        dropout: float,
    ):
        super().__init__()
        self.num_patch = (img_size**2) // (patch_size**2)
        _assert(
            isinstance(self.num_patch, int),
            "Number of patches derived from image size and patch size is not an integer",
        )

        self.patch_embedding = PatchEmbedding(
            in_channel=in_channel, emb_size=emb_size, patch_size=patch_size
        )
        self.cls_token = nn.Parameter(rand(1, 1, emb_size))
        self.position_embedding = nn.Parameter(rand(1, self.num_patch + 1, emb_size))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, img):
        x = self.patch_embedding(img)
        b = x.shape[0]
        cls_token = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = cat([x, cls_token], dim=1)
        x += self.position_embedding
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.att = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads, dropout=dropout
        )
        self.qkv = nn.Linear(emb_dim, emb_dim * 3, bias=False)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dims=-1)
        return self.att(q, k, v)


class MLP(MLP):
    def __init__(self, in_dim, mlp_dim, dropout):
        super().__init__(
            in_channels=in_dim, hidden_channels=[in_dim, mlp_dim], dropout=dropout
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x = x + res
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.att = Attention(emb_dim=emb_size, num_heads=num_heads, dropout=dropout)
        self.mlp = MLP(in_dim=emb_size, mlp_dim=emb_size, dropout=dropout)
        self.attResidualAdd = ResidualAdd(PreNorm(emb_size, self.att))
        self.mlpResidualAdd = ResidualAdd(PreNorm(emb_size, self.mlp))

    def forward(self, x):
        x = self.attResidualAdd(x)
        x = self.mlpResidualAdd(x)
        return x


# class PreNorm(nn.Module):
#     def __init__(self):
#         pass
#
#     def forward(self, x):
#         pass

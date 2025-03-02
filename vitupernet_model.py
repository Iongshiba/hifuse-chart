from torch import alias_copy, align_tensors, nn, cat, rand, _assert, softmax
import torch.nn.functional as F
from torch.nn.modules import BatchNorm2d
from torchvision.ops.misc import MLP
from einops.layers.torch import Rearrange
from einops import repeat


# COULD USE HIFUSE PATCHEMBEDDING MODULE
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channel: int = 3,
        emb_size: int = 128,
        patch_size: int = 16,
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


class PyramidPoolingModule(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int = 256, pool_sizes=[1, 2, 3, 6]
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_sizes = pool_sizes

        self.avg_pooled = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=in_channels // 4,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(num_features=in_channels // 4),
                    nn.ReLU(inplace=True),
                )
                for pool_size in pool_sizes
            ]
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        pooled_layers = [
            F.interpolate(layer(x), size=(h, w), mode="bilinear", align_corners=False)
            for layer in self.avg_pooled
        ]

        x = cat([x] + pooled_layers, dim=1)

        return self.final_conv(x)


class ViTUperNet(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        num_class=4,
        in_channels=3,
        out_channels=256,
        embed_dim=1024,
        num_trans=(4, 6, 8, 6),
        num_heads=(16, 16, 16, 16),
        dropout=0.1,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()

        self.num_patch = img_size // patch_size
        self.C = patch_size**2 * in_channels

        self.embeddings = Embedding(
            in_channel=in_channels,
            img_size=img_size,
            emb_size=embed_dim,
            patch_size=patch_size,
            dropout=dropout,
        )

        self.transformers = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        TransformerBlock(
                            emb_size=embed_dim, num_heads=num_head, dropout=dropout
                        )
                        for _ in range(num_tran)
                    ]
                )
                for num_head, num_tran in zip(num_heads, num_trans)
            ]
        )
        self.ppm = PyramidPoolingModule(
            in_channels=embed_dim, out_channels=out_channels
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv1x1 = nn.Conv2d(
            in_channels=embed_dim, out_channels=out_channels, kernel_size=1
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3
        )
        self.conv3x3fusion = nn.Conv2d(
            in_channels=embed_dim, out_channels=out_channels, kernel_size=3
        )

        self.stage1Upscale = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.C,
                out_channels=embed_dim,
                kernel_size=2,
                stride=2,
                bias=False,
            ),
            norm_layer(out_channels),
            nn.GELU(),
            nn.ConvTranspose2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=2,
                stride=2,
                bias=False,
            ),
        )
        self.stage2Upscale = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.C,
                out_channels=embed_dim,
                kernel_size=2,
                stride=2,
                bias=False,
            ),
        )
        self.stage4Downscale = nn.Sequential(nn.MaxPool2d(kernel_size=2))

        self.head = nn.Softmax(dim=num_class)

    def forward(self, img):
        # 1 Image Input: Embeddings (b, c, h, w) -> (b (h/16 w/16) (p p c))
        x = self.embeddings(img)

        # 2 ViT Feature Extraction
        stage_1 = self.transformers[0](x)
        stage_2 = self.transformers[1](stage_1)
        stage_3 = self.transformers[2](stage_2)
        stage_4 = self.transformers[3](stage_3)

        upscaled_stage_1 = self.stage1Upscale(stage_1)
        upscaled_stage_2 = self.stage2Upscale(stage_2)
        downscaled_stage_4 = self.stage4Downscale(stage_4)

        # 3 PPM
        p4 = self.ppm(self.max_pool(downscaled_stage_4))  # (b, h/32, w/32, 256)

        # 4 FPN
        upscaled_p4 = F.interpolate(
            input=p4, scale_factor=2, mode="bilinear", align_corners=False
        )
        p3 = self.conv3x3(
            F.relu(upscaled_p4 + self.conv1x1(stage_3))
        )  # (b, h/16, w/16, 256)

        upscaled_p3 = F.interpolate(
            input=p3, scale_factor=2, mode="bilinear", align_corners=False
        )
        p2 = self.conv3x3(
            F.relu(upscaled_p3 + self.conv1x1(upscaled_stage_2))
        )  # (b, h/8, w/8, 256)

        upscaled_p2 = F.interpolate(
            input=p2, scale_factor=2, mode="bilinear", align_corners=False
        )
        p1 = self.conv3x3(
            F.relu(upscaled_p2 + self.conv1x1(upscaled_stage_1))
        )  # (b, h/4, w/4, 256)

        # 5 Feature fusion

        fusion_p4, fusion_p3, fusion_p2, fusion_p1 = (
            F.interpolate(
                input=p,
                size=(self.img_size // 4, self.img_size // 4),
                align_corners=False,
            )
            for p in [p4, p3, p2, p1]
        )

        fusion = cat([fusion_p4, fusion_p3, fusion_p2, fusion_p1], dim=1)
        fusion = self.conv3x3fusion(fusion)
        fusion = F.interpolate(
            input=fusion,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        # 6 Output
        return self.head(fusion)

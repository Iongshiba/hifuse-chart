import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import (_assert, alias_copy, align_tensors, cat, nn, rand,
                   return_types, softmax)
from torch.nn.modules import BatchNorm2d
from torchvision.ops.misc import MLP


# COULD USE HIFUSE PATCHEMBEDDING MODULE
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channel: int = 3,
        emb_size: int = 1024,
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
        self.num_patch = (img_size // patch_size) ** 2
        _assert(
            isinstance(self.num_patch, int),
            "Number of patches derived from image size and patch size is not an integer",
        )

        self.patch_embedding = PatchEmbedding(
            in_channel=in_channel, emb_size=emb_size, patch_size=patch_size
        )
        # self.cls_token = nn.Parameter(rand(1, 1, emb_size))
        self.position_embedding = nn.Parameter(rand(1, self.num_patch, emb_size))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, img):
        x = self.patch_embedding(img)
        # b = x.shape[0]
        # cls_token = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        # x = cat([cls_token, x], dim=1)
        # print(self.position_embedding.shape)

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
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        output, _ = self.att(q, k, v)
        return output


class MLP(MLP):
    def __init__(self, in_dim, mlp_dim, dropout):
        super().__init__(
            in_channels=in_dim, hidden_channels=[in_dim, mlp_dim], dropout=dropout
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x = x + res
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
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
                        out_channels=out_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                )
                for pool_size in pool_sizes
            ]
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels * 2,
                out_channels=out_channels,
                kernel_size=3,
                bias=False,
            ),
        )

    def forward(self, x):
        # print(x.shape)
        h, w = x.shape[2], x.shape[3]
        # print(h.shape, w.shape)
        pooled_layers = [
            F.interpolate(layer(x), size=(h, w), mode="bilinear", align_corners=False)
            for layer in self.avg_pooled
        ]

        x = cat([x] + pooled_layers, dim=1)
        print("dit me may: ", x.shape)

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

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patch = img_size // patch_size

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
        # self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv1x1 = nn.Conv2d(
            in_channels=embed_dim, out_channels=out_channels, kernel_size=1
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv3x3fusion = nn.Conv2d(
            in_channels=embed_dim, out_channels=out_channels, kernel_size=3
        )

        self.stage1Upscale = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=2,
                stride=2,
                bias=False,
            ),
            norm_layer([embed_dim, self.num_patch * 2, self.num_patch * 2]),
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
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=2,
                stride=2,
                bias=False,
            ),
        )
        self.stage4Downscale = nn.Sequential(nn.MaxPool2d(kernel_size=2))

        self.final_conv = nn.Conv2d(
            in_channels=out_channels, out_channels=num_class, kernel_size=1
        )

    def forward(self, img):
        # 1 Image Input: Embeddings (b, c, h, w) -> (b (h/16 w/16) (p p c))
        x = self.embeddings(img)

        # 2 ViT Feature Extraction
        stage_1 = self.transformers[0](x)
        stage_2 = self.transformers[1](stage_1)
        stage_3 = self.transformers[2](stage_2)
        stage_4 = self.transformers[3](stage_3)
        # print("Stage 1 shape: ", stage_1.shape)
        # print("Stage 2 shape: ", stage_2.shape)
        # print("Stage 3 shape: ", stage_3.shape)
        # print("Stage 4 shape: ", stage_4.shape)

        stage_1 = rearrange(
            stage_1,
            "b (h w) d -> b d h w",
            h=self.num_patch,
            w=self.num_patch,
        )  # (b, n, d) -> (b, d, h, w)
        upscaled_stage_1 = self.stage1Upscale(stage_1)

        stage_2 = rearrange(
            stage_2,
            "b (h w) d -> b d h w",
            h=self.num_patch,
            w=self.num_patch,
        )
        upscaled_stage_2 = self.stage2Upscale(stage_2)

        stage_4 = rearrange(
            stage_4,
            "b (h w) d -> b d h w",
            h=self.num_patch,
            w=self.num_patch,
        )
        downscaled_stage_4 = self.stage4Downscale(stage_4)

        # print("Upscaled stage 1 shape: ", upscaled_stage_1.shape)
        # print("Upscaled stage 2 shape: ", upscaled_stage_2.shape)
        print("Downscaled stage 4 shape: ", downscaled_stage_4.shape)

        # 3 PPM
        p4 = self.ppm(downscaled_stage_4)  # (b, h/32, w/32, 256)

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
        fusion_p4 = F.interpolate(
            input=p4, scale_factor=8, mode="bilinear", align_corners=False
        )
        fusion_p3 = F.interpolate(
            input=p3, scale_factor=4, mode="bilinear", align_corners=False
        )
        fusion_p2 = F.interpolate(
            input=p2, scale_factor=2, mode="bilinear", align_corners=False
        )
        fusion = cat([fusion_p4, fusion_p3, fusion_p2, p1], dim=1)
        fusion = self.conv3x3fusion(fusion)
        fusion = F.interpolate(
            input=fusion,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        # 6 Output
        fusion = self.final_conv(fusion)
        return F.softmax(fusion, dim=1)

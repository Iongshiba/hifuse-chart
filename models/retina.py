import torch
import torch.nn as nn
import torch.nn.functional as F


class Retina(nn.Module):
    """
    RetinaNet Head for object detection.

    This module consists of classification and regression subnets applied to feature maps
    from a Feature Pyramid Network (FPN). It supports optional feature map fusion to generate
    a single prediction instead of separate predictions for each level of the FPN.

    Attributes:
        fuse_fm (bool): If True, fuses feature maps to a unified size before making predictions.
        cls_subnet (nn.Sequential): Convolutional layers for the classification branch.
        cls_logits (nn.Conv2d): Final classification layer predicting class logits per anchor.
        reg_subnet (nn.Sequential): Convolutional layers for the bounding box regression branch.
        bbox_reg (nn.Conv2d): Final regression layer predicting bounding box offsets per anchor.
        fuse (nn.Conv2d): 1x1 convolution for fusing upsampled feature maps (used if `fuse_fm=True`).

    Args:
        num_classes (int): Number of object classes to detect.
        fuse_fm (bool, optional): Whether to fuse feature maps before prediction. Defaults to True.
        num_fm (int, optional): Number of input feature maps. Defaults to 4.
        in_channels_list (list[int], optional): List of input channels for each FPN level. Defaults to [96, 192, 384, 768].
        num_anchors (int, optional): Number of anchor boxes per feature map location. Defaults to 9.
        out_channels (int, optional): Number of output channels for the feature maps. Defaults to 192.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels_list: list[int] = [96, 192, 384, 768],
        fuse_fm: bool = True,
        num_fm: int = 4,
        num_anchors: int = 9,
        out_channels: int = 192,
    ):
        super().__init__()

        self.fuse_fm = fuse_fm
        self.num_fm = num_fm

        # 1x1 Convolutions to unify channels
        self.channel_align = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
                for in_channels in []
            ]
        )

        # Classification branch
        self.cls_subnet = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cls_logits = nn.Conv2d(
            out_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )

        # Regression branch
        self.reg_subnet = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.bbox_reg = nn.Conv2d(
            out_channels, num_anchors * num_fm, kernel_size=3, padding=1
        )

        # Fuse the feature maps
        self.fuse = nn.Conv2d(num_fm * out_channels, out_channels, kernel_size=1)

    def forward(self, feature_maps):
        # Assert if the `num_fm` matches with the actual length of `feature_maps`
        assert (
            len(feature_maps) == self.num_fm
        ), "The number of feature maps differs from the `num_fm` attribute of the class"

        # Align feature maps to have `out_channels` channels
        feature_maps = [
            self.channel_align[i](feature) for i, feature in enumerate(feature_maps)
        ]

        # Fuse the feature maps to highest size (56x56x`out_channels`) and return 1 prediction
        if self.fuse_fm:
            # Get the highest size (56x56)
            h, w = (
                feature_maps[0].shape[2],
                feature_maps[0].shape[3],
            )

            # Upsample all feature maps to 56x56
            upsampled_maps = [
                F.interpolate(fm, size=(h, w), mode="bilinear", align_corners=False)
                for fm in feature_maps
            ]

            # Concatenate along channel dimension -> (b, c, h, w)
            fused_feature = torch.cat(upsampled_maps, dim=1)
            print(fused_feature.shape)
            fused_feature = self.fuse(fused_feature)

            cls_preds = self.cls_logits(self.cls_subnet(fused_feature))
            reg_preds = self.bbox_reg(self.reg_subnet(fused_feature))

        else:  # Don't fuse, return `num_fm` predictions
            cls_preds = []
            reg_preds = []

            for feature in feature_maps:
                cls_out = self.cls_logits(self.cls_subnet(feature))
                reg_out = self.bbox_reg(self.reg_subnet(feature))

                cls_preds.append(cls_out)
                reg_preds.append(reg_out)

        return cls_preds, reg_preds

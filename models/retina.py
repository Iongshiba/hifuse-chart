import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torchvision.ops as ops
from torchvision.ops import batched_nms, boxes as box_ops
import torchvision.models.detection._utils as det_utils
from torchvision.models.detection.retinanet import _sum, _box_loss


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
        in_channels_list: list[int] = [32, 64, 128, 256],
        out_channels: int = 256,
        fuse_fm: bool = True,
        num_fm: int = 4,
        num_anchors: int = 9,
        anchor_generator=None,
        proposal_matcher=None,
    ):
        super().__init__()

        self.fuse_fm = fuse_fm
        self.num_fm = num_fm
        if proposal_matcher is None:
            proposal_matcher = self._default_proposal_matcher(0.4, 0.5, True)
        self.proposal_matcher = proposal_matcher
        self.BETWEEN_THRESHHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        if anchor_generator:
            self.anchor_generator = anchor_generator
        else:
            self.anchor_generator = self._default_anchor_gen()

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

    def forward(self, feature_maps, images, targets):
        # Assert if the `num_fm` matches with the actual length of `feature_maps`
        assert len(feature_maps) == self.num_fm, (
            "The number of feature maps differs from the `num_fm` attribute of the class"
        )

        if self.fuse_fm:  # Fuse feature maps into (56x56x`out_channels`)
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

            # Concatenate along channel dimension (b, c, h, w)
            fused_feature = torch.cat(upsampled_maps, dim=1)
            fused_feature = self.fuse(fused_feature)

            cls_logits = [self.cls_logits(self.cls_subnet(fused_feature))]
            reg_logits = [self.bbox_reg(self.reg_subnet(fused_feature))]

        else:  # No fuse, keep each feature map separate
            cls_logits = []
            reg_logits = []

            for feature in feature_maps:
                cls_out = self.cls_logits(self.cls_subnet(feature))
                reg_out = self.bbox_reg(self.reg_subnet(feature))

                cls_logits.append(cls_out)
                reg_logits.append(reg_out)

        image_list = self._to_ImageList(images)
        anchors = self.anchor_generator(image_list, feature_maps)

        # If model training -> return loss
        if self.training:
            assert targets is not None, (
                "during training, targets of the images are required"
            )

            return self.compute_loss(targets, cls_logits, reg_logits, anchors)

        else:  # inference (return prediction dict of cls and bbox)
            return self.postprocess(cls_logits, reg_logits, images, anchors)

    def _to_ImageList(self, images):
        original_sizes = [img.shape[-2:] for img in images]
        return ImageList(images, original_sizes)

    def _default_proposal_matcher(
        self, fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True
    ):
        proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches,
        )
        return proposal_matcher

    def _default_anchor_gen(self):
        sizes = (
            (16, 32, 64),  # P1 (56x56)
            (32, 64, 128),  # P2 (28x28)
            (64, 128, 192),  # P3 (14x14)
            (128, 192, 224),  # P4 (7x7)
        )

        anchor_generator = AnchorGenerator(
            sizes=sizes,
            aspect_ratios=((0.5, 1.0, 2.0),) * 4,
        )
        return anchor_generator

    def compute_loss(self, targets, cls_logits, bbox_reg, anchors):
        matched_idxs = self.match_anchors(targets, anchors)

        # Classification loss
        cls_loss = self._cls_loss(targets, cls_logits, matched_idxs)

        # BBox Regression loss
        reg_loss = self._reg_loss(targets, bbox_reg, anchors, matched_idxs)

        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
        }

    # PyTorch RetinaNetClassificationHead compute_loss()
    def _cls_loss(self, targets, cls_logits, matched_idxs):
        losses = []

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(
            targets, cls_logits, matched_idxs
        ):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][
                    matched_idxs_per_image[foreground_idxs_per_image]
                ],
            ] = 1.0

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHHOLDS

            # compute the class classification loss
            losses.append(
                sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    reduction="sum",
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / len(targets)

    # PyTorch RetinaNetRegressionHead compute_loss()
    def _reg_loss(self, targets, bbox_reg, anchors, matched_idxs):
        losses = []

        for (
            targets_per_image,
            bbox_reg_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, bbox_reg, anchors, matched_idxs):
            # determine only the forground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image["boxes"][
                foreground_idxs_per_image, :
            ]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the loss
            losses.append(
                _box_loss(
                    "l1",
                    self.box_coder,
                    anchors_per_image,
                    matched_gt_boxes_per_image,
                    bbox_reg_per_image,
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / max(1, len(targets))

    def postprocess(
        self, cls_logits, reg_logits, images, anchors, score_thresh=0.5, nms_thresh=0.5
    ):
        results = []

        for img_idx in range(len(images)):
            cls_logits_per_image = [l[img_idx] for l in cls_logits]
            reg_logits_per_image = [l[img_idx] for l in reg_logits]
            anchors_per_image = anchors[img_idx]

            cls_scores = torch.cat(
                [
                    l.permute(1, 2, 0).reshape(-1, self.num_classes)
                    for l in cls_logits_per_image
                ],
                dim=0,
            )
            reg_deltas = torch.cat(
                [l.permute(1, 2, 0).reshape(-1, 4) for l in reg_logits_per_image], dim=0
            )

            cls_probs = cls_scores.sigmoid()
            boxes = self.box_coder.decode(reg_deltas, anchors_per_image)

            max_scores, labels = cls_probs.max(dim=1)
            keep = max_scores > score_thresh
            boxes, scores, labels = boxes[keep], max_scores[keep], labels[keep]

            keep_idxs = batched_nms(boxes, scores, labels, nms_thresh)
            boxes, scores, labels = (
                boxes[keep_idxs],
                max_scores[keep_idxs],
                labels[keep_idxs],
            )

            results.append({"boxes": boxes, "scores": scores, "labels": labels})

        return results

    def match_anchors(self, targets, anchors):
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            # No ground truth check
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full(
                        (anchors_per_image.size(0),),
                        -1,
                        dtype=torch.int64,
                        device=anchors_per_image.device,
                    )
                )
                continue

            match_quality_matrix = box_ops.box_iou(
                targets_per_image["boxes"], anchors_per_image
            )
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))
        return matched_idxs

import math
from typing import List, Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.models.detection._utils as det_utils
from torchvision.ops import clip_boxes_to_image
from torchvision.models.detection.retinanet import _sum
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import batched_nms
from torchvision.ops import boxes as box_ops
from torchvision.ops.focal_loss import sigmoid_focal_loss


class RetinaNetHead(nn.Module):
    """
    RetinaNet classification and regression heads
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        num_convs: int = 4,
    ):
        """
        Initialize RetinaNet heads

        Args:
            in_channels: Number of input channels from FPN
            num_anchors: Number of anchors per location
            num_classes: Number of object classes to predict
            num_convs: Number of convolutional layers in each subnet. Defaults to 4 as stated in the paper
        """
        super().__init__()

        self.num_classes = num_classes

        # Classification subnet
        cls_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet.append(nn.ReLU(inplace=True))
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )

        # Box regression subnet
        bbox_subnet = []
        for _ in range(num_convs):
            bbox_subnet.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            bbox_subnet.append(nn.ReLU(inplace=True))
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )

        # probability initialization for classification subnet
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - 0.01) / 0.01))

        # Normal initialization for bbox regression subnet
        nn.init.normal_(self.bbox_pred.weight, std=0.01)
        nn.init.constant_(self.bbox_pred.bias, 0)

        for modules in [self.cls_subnet, self.bbox_subnet]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out", nonlinearity="relu"
                    )
                    nn.init.constant_(layer.bias, 0)

    def forward(self, features: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features: List of feature maps from FPN

        Returns:
            Tuple containing:
            - cls_logits: Classification logits for each anchor
            - bbox_pred: Bounding box regression values for each anchor
        """
        # Apply the subnets to each feature level
        cls_logits = []
        bbox_preds = []

        for feature in features:
            # Classification branch
            cls_output = self.cls_logits(self.cls_subnet(feature))

            # Reshape classification output: (N, A*C, H, W) -> (N, H*W*A, C)
            N, _, H, W = cls_output.shape
            cls_output = cls_output.view(N, -1, self.num_classes, H, W)
            cls_output = cls_output.permute(0, 3, 4, 1, 2)
            cls_output = cls_output.reshape(N, -1, self.num_classes)
            cls_logits.append(cls_output)

            # Regression branch
            bbox_output = self.bbox_pred(self.bbox_subnet(feature))

            # Reshape bbox output: (N, A*4, H, W) -> (N, H*W*A, 4)
            N, _, H, W = bbox_output.shape
            bbox_output = bbox_output.view(N, -1, 4, H, W)
            bbox_output = bbox_output.permute(0, 3, 4, 1, 2)
            bbox_output = bbox_output.reshape(N, -1, 4)
            bbox_preds.append(bbox_output)

        # Concatenate outputs
        return torch.cat(cls_logits, dim=1), torch.cat(bbox_preds, dim=1)


class RetinaNet(nn.Module):
    """
    RetinaNet object detection model

    A one-stage detector using Feature Pyramid Network (FPN) for feature extraction
    and Focal Loss to address class imbalance problem.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels_list: List[int] = None,
        out_channels: int = 256,
        fuse_fm: bool = False,
        num_anchors: int = 9,
        anchor_generator=None,
        proposal_matcher: det_utils.Matcher = None,
        box_loss_weight: float = 1.0,
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
        bbox_reg_loss_type: str = "smooth_l1",
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 100,
    ):
        """
        Initialize RetinaNet model

        Args:
            num_classes: Number of object classes (excluding background)
            in_channels_list: List of channel dimensions for backbone features
            out_channels: Number of channels in FPN and prediction heads
            anchor_generator: Optional custom AnchorGenerator
            anchor_sizes: Sizes of anchors per feature level
            aspect_ratios: Aspect ratios of anchors at each feature level
            proposal_matcher: Matcher for assigning ground truth to anchors
            box_loss_weight: Weight of bounding box regression loss
            focal_loss_alpha: Alpha parameter for focal loss
            focal_loss_gamma: Gamma parameter for focal loss
            bbox_reg_loss_type: Type of bounding box regression loss
            score_thresh: Threshold for filtering low confidence predictions
            nms_thresh: IoU threshold for NMS
            detections_per_img: Maximum number of detections per image
        """
        super().__init__()

        self.fuse_fm = fuse_fm

        num_anchors = num_anchors
        self.head = RetinaNetHead(
            in_channels=out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

        # Anchor generator
        if anchor_generator is None:
            anchor_sizes = (
                (14, 28, 56),  # P1 (56x56)
                (28, 56, 112),  # P2 (28x28)
                (56, 112, 224),  # P3 (14x14)
                (112, 224, 448),  # P4 (7x7)
            )

            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

            anchor_generator = AnchorGenerator(
                sizes=anchor_sizes,
                aspect_ratios=aspect_ratios,
            )

        self.anchor_generator = anchor_generator

        # Proposal matcher
        if proposal_matcher is None:
            proposal_matcher = det_utils.Matcher(
                high_threshold=0.5, low_threshold=0.4, allow_low_quality_matches=True
            )

        self.proposal_matcher = proposal_matcher
        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

        # Box coder for converting between formats
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # Loss parameters
        self.box_loss_weight = box_loss_weight
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.bbox_reg_loss_type = bbox_reg_loss_type

        # Inference parameters
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        # Num of classes
        self.num_classes = num_classes

        # Fuse
        self.fuse = nn.Conv2d(4 * out_channels, out_channels, kernel_size=1)

    def forward(
        self,
        feature_maps: list[Tensor],
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Forward pass

        Args:
            images: List of input images (Tensors)
            targets: Optional list of dictionaries with keys:
                    'boxes': Tensor of ground truth boxes (x1, y1, x2, y2)
                    'labels': Tensor of ground truth labels

        Returns:
            During training: Dict of losses
            During inference: List of dicts with 'boxes', 'scores', 'labels'

        Example flow of data:
            1. images: [tensor(3, H, W), ...] -> feature maps from TriFuse
            2. feature maps -> head -> cls_logits, bbox_pred
            3. Generate anchors from feature maps
            4. Training:
               - Match anchors to ground truth targets
               - Compute losses
            5. Inference:
               - Apply regression to anchors
               - Filter predictions
               - Apply NMS
        """
        # Validate inputs
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed in")

        # Fuse feature maps
        # if self.fuse_fm:
        #     h, w = (feature_maps[0].shape[2], feature_maps[0].shape[3])

        #     # Upsample all feature maps to 56x56
        #     upsampled_maps = [
        #         F.interpolate(fm, size=(h, w), mode="bilinear", align_corners=False)
        #         for fm in feature_maps
        #     ]

        #     # Concatenate along channel dimension (b, c, h, w)
        #     feature_maps = torch.cat(upsampled_maps, dm=1)
        #     feature_maps = [self.fuse(feature_maps)]

        # Convert images to ImageList
        original_image_sizes = [img.shape[-2:] for img in images]
        image_list = ImageList(images, original_image_sizes)

        # Feed through RetinaNetHead to get logits
        cls_logits, bbox_pred = self.head(feature_maps)

        # Generate anchors for each feature map level
        anchors = self.anchor_generator(image_list, feature_maps)
        anchors = [
            clip_boxes_to_image(anchor, image_size)
            for anchor, image_size in zip(anchors, image_list.image_sizes)
        ]

        # If training, return losses
        if self.training:
            return self._compute_loss(targets, cls_logits, bbox_pred, anchors)

        # Otherwise, perform inference
        else:
            return self._inference(
                cls_logits, bbox_pred, anchors, image_list.image_sizes
            )

    def _compute_loss(
        self,
        targets: List[Dict[str, Tensor]],
        cls_logits: Tensor,
        bbox_pred: Tensor,
        anchors: List[Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute RetinaNet losses

        Args:
            targets: Ground truth targets
            cls_logits: Classification logits for all anchors
            bbox_pred: Bounding box regression for all anchors
            anchors: Anchor boxes

        Returns:
            Dictionary of classification and regression losses

        Example flow:
            1. Match anchors to ground truth using IoU
            2. Compute focal loss for classification
            3. Compute smooth L1 loss for box regression
            4. Return weighted sum of losses
        """
        # Match targets to anchors
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                # No ground truth boxes, all anchors are background
                matched_idxs.append(
                    torch.full(
                        (anchors_per_image.size(0),),
                        -1,
                        dtype=torch.int64,
                        device=anchors_per_image.device,
                    )
                )
                continue

            # Compute IoU between ground truth boxes and anchors
            match_quality_matrix = box_ops.box_iou(
                targets_per_image["boxes"], anchors_per_image
            )

            # Match each anchor to the ground truth box with highest IoU
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        # Compute classification loss
        cls_loss = self._classification_loss(targets, cls_logits, matched_idxs)

        # Compute box regression loss
        box_loss = self._regression_loss(targets, bbox_pred, anchors, matched_idxs)

        # Total loss is a weighted sum of classification and regression losses
        return {
            "cls_loss": cls_loss,
            "box_loss": box_loss,
            "loss": cls_loss + self.box_loss_weight * box_loss,
        }

    def _classification_loss(
        self,
        targets: List[Dict[str, Tensor]],
        cls_logits: Tensor,
        matched_idxs: List[Tensor],
    ) -> Tensor:
        """
        Compute focal loss for classification

        Args:
            targets: Ground truth targets
            cls_logits: Classification logits for all anchors
            matched_idxs: Indices matching anchors to ground truth

        Returns:
            Focal loss for classification

        Flow:
            1. For each image, identify foreground anchors
            2. Create target tensor with one-hot encoding
            3. Compute focal loss between logits and targets
        """
        losses = []

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(
            targets, cls_logits, matched_idxs
        ):
            cls_logits_per_image = cls_logits_per_image.view(
                cls_logits_per_image.shape[1], -1
            ).T

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
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the class classification loss
            losses.append(
                sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="sum",
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / len(targets)

    def _regression_loss(
        self,
        targets: List[Dict[str, Tensor]],
        bbox_regression: Tensor,
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ) -> Tensor:
        """
        Compute regression loss for bounding boxes

        Args:
            targets: Ground truth targets
            bbox_regression: Box regression values for all anchors
            anchors: Anchor boxes
            matched_idxs: Indices matching anchors to ground truth

        Returns:
            Box regression loss

        Example:
            1. For each image, identify foreground anchors
            2. Extract corresponding ground truth boxes
            3. Compute regression loss
        """
        losses = []

        for (
            targets_per_image,
            bbox_reg_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, bbox_regression, anchors, matched_idxs):
            # determine only the forground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image["boxes"][
                matched_idxs_per_image[foreground_idxs_per_image]
            ]
            bbox_reg_per_image = bbox_reg_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            if matched_gt_boxes_per_image.numel() == 0:
                matched_gt_boxes_per_image = torch.full(
                    (1, 4),
                    -1.0,
                    dtype=torch.float32,
                    device=matched_gt_boxes_per_image.device,
                )

            # compute the loss
            losses.append(
                det_utils._box_loss(
                    self.bbox_reg_loss_type,
                    self.box_coder,
                    anchors_per_image,
                    matched_gt_boxes_per_image,
                    bbox_reg_per_image,
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / max(1, len(targets))

    def _inference(
        self,
        cls_logits: List[Tensor],
        bbox_regression: List[Tensor],
        anchors: List,
        image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:

        # --- 0. make sure anchors is List[images][levels] -> Tensor[level_i,4]
        if isinstance(anchors[0], torch.Tensor) and anchors[0].dim() == 3:
            B = anchors[0].size(0)
            anchors = [
                [lvl[i] for lvl in anchors] for i in range(B)  # for each image i
            ]

        def _ensure_2d(x: Tensor) -> Tensor:
            return x.unsqueeze(0) if x.dim() == 1 else x

        detections: List[Dict[str, Tensor]] = []
        num_images = len(image_sizes)

        for img_idx in range(num_images):
            # grab per-image, per-level tensors
            box_regs = [br[img_idx] for br in bbox_regression]
            logits = [cl[img_idx] for cl in cls_logits]
            img_anchors, img_shape = anchors[img_idx], image_sizes[img_idx]

            all_boxes, all_scores, all_labels = [], [], []

            for br_lvl, logit_lvl, anch_lvl in zip(box_regs, logits, img_anchors):
                # br_lvl:  (N_anchors, 4)
                # logit_lvl: (N_anchors, num_classes)
                # anch_lvl:   (N_anchors, 4)

                num_classes = logit_lvl.shape[-1]
                scores = torch.sigmoid(logit_lvl).flatten()
                keep = scores > self.score_thresh

                if not keep.any():
                    continue

                scores = scores[keep]
                idxs = torch.where(keep)[0]

                # top-K
                K = det_utils._topk_min(idxs, self.detections_per_img, 0)
                scores, topk_idx = scores.topk(K)
                idxs = idxs[topk_idx]

                # map flat idx -> anchor idx + label
                anchor_idxs = torch.div(idxs, num_classes, rounding_mode="floor")
                labels = idxs % num_classes

                # slice out the regressions and anchors
                sel_regs = _ensure_2d(br_lvl[anchor_idxs])
                sel_anc = _ensure_2d(anch_lvl[anchor_idxs])

                # decode + clip
                boxes = self.box_coder.decode_single(sel_regs, sel_anc)
                boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

            if not all_scores:
                # no detections for this image
                detections.append(
                    {
                        "boxes": torch.zeros(0, 4),
                        "scores": torch.zeros(0),
                        "labels": torch.zeros(0, dtype=torch.int64),
                    }
                )
                continue

            boxes = torch.cat(all_boxes, 0)
            scores = torch.cat(all_scores, 0)
            labels = torch.cat(all_labels, 0)

            # final NMS and cap at top detection count
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": boxes[keep],
                    "scores": scores[keep],
                    "labels": labels[keep],
                }
            )

        return detections

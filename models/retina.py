import math
from typing import List, Tuple, Dict, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.models.detection._utils as det_utils
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
                    nn.init.normal_(layer.weight, std=0.01)
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


class Retina(nn.Module):
    """
    RetinaNet object detection model

    A one-stage detector using Feature Pyramid Network (FPN) for feature extraction
    and Focal Loss to address class imbalance problem.
    """

    def __init__(
        self,
        num_classes: int,
        # Feature maps parameters
        in_channels_list: List[int] = None,
        out_channels: int = 256,
        fuse_fm: bool = False,
        # Anchor parameters
        num_anchors: int = 9,
        anchor_generator: AnchorGenerator = None,
        anchor_sizes: Tuple[Tuple[int, ...]] = None,
        aspect_ratios: List[Tuple[float, ...]] = None,
        # Other parameters
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
            if anchor_sizes is None:
                anchor_sizes = (
                    (16, 32, 64),  # P1 (56x56)
                    (32, 64, 128),  # P2 (28x28)
                    (64, 128, 192),  # P3 (14x14)
                    (128, 192, 224),  # P4 (7x7)
                )

            if aspect_ratios is None:
                aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

            anchor_generator = AnchorGenerator(
                sizes=anchor_sizes,
                aspect_ratios=aspect_ratios,
            )

        self.anchor_generator = anchor_generator

        # Proposal matcher
        if proposal_matcher is None:
            proposal_matcher = det_utils.Matcher(
                high_threshold=0.5, low_threshold=0.4, allow_low_quality_matches=False
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

    def forward(
        self,
        feature_maps: list[Tensor],
        images: List[Tensor],
        targets: List[Dict[str, Tensor]] = None,
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
        if self.fuse_fm:
            h, w = (feature_maps[0].shape[2], feature_maps[0].shape[3])

            # Upsample all feature maps to 56x56
            upsampled_maps = [
                F.interpolate(fm, size=(h, w), mode="bilinear", align_corners=False)
                for fm in feature_maps
            ]

            # Concatenate along channel dimension (b, c, h, w)
            feature_maps = torch.cat(upsampled_maps, dim=1)
            feature_maps = [self.fuse(feature_maps)]

        # Convert images to ImageList
        original_image_sizes = [img.shape[-2:] for img in images]
        image_list = ImageList(torch.stack(images), original_image_sizes)

        # Feed through RetinaNetHead to get logits
        cls_logits, bbox_pred = self.head(feature_maps)

        # Generate anchors for each feature map level
        anchors = self.anchor_generator(image_list, feature_maps)

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
            # Identify foreground (matched to a ground truth box)
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # Create one-hot encoded target tensor. Shape: [num_anchors, num_classes]
            gt_classes_target = torch.zeros_like(cls_logits_per_image)

            # Only set values for foreground anchors
            if num_foreground > 0:
                # Get ground truth labels for matched anchors
                # matched_idxs_per_image[foreground_idxs_per_image] gives the indices of ground truth boxes that were matched to each foreground anchor
                gt_classes_idxs = targets_per_image["labels"][
                    matched_idxs_per_image[foreground_idxs_per_image]
                ]

                # Set the corresponding class to 1 (one-hot encoding)
                gt_classes_target[foreground_idxs_per_image, gt_classes_idxs] = 1.0

            # Identify valid anchors (not between thresholds)
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # Compute focal loss
            if valid_idxs_per_image.sum() > 0:
                losses.append(
                    sigmoid_focal_loss(
                        cls_logits_per_image[valid_idxs_per_image],
                        gt_classes_target[valid_idxs_per_image],
                        alpha=self.focal_loss_alpha,
                        gamma=self.focal_loss_gamma,
                        reduction="sum",
                    )
                    / max(
                        1, num_foreground
                    )  # Normalize by number of foreground anchors
                )
            else:
                losses.append(cls_logits_per_image.sum() * 0)  # Empty loss

        # Average loss across all images in batch
        return torch.stack(losses).mean()

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
            bbox_regression_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, bbox_regression, anchors, matched_idxs):
            # Find foreground anchors
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # Skip if no foreground anchors
            if num_foreground == 0:
                losses.append(bbox_regression_per_image.sum() * 0)
                continue

            # Get ground truth boxes for foreground anchors
            matched_gt_boxes = targets_per_image["boxes"][
                matched_idxs_per_image[foreground_idxs_per_image]
            ]

            # Get predicted boxes for foreground anchors
            bbox_regression_per_image = bbox_regression_per_image[
                foreground_idxs_per_image, :
            ]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # Encode ground truth boxes as targets for regression
            # (convert from absolute coordinates to offsets relative to anchors)
            target_regression = self.box_coder.encode(
                matched_gt_boxes, anchors_per_image
            )

            # Compute loss based on specified regression loss type
            if self.bbox_reg_loss_type == "smooth_l1":
                loss = F.smooth_l1_loss(
                    bbox_regression_per_image,
                    target_regression,
                    beta=1.0 / 9.0,  # As used in Faster R-CNN
                    reduction="sum",
                )
            elif self.bbox_reg_loss_type == "giou":
                # Decode predictions to boxes
                pred_boxes = self.box_coder.decode(
                    bbox_regression_per_image, anchors_per_image
                )
                loss = 1 - torch.diag(
                    box_ops.generalized_box_iou(pred_boxes, matched_gt_boxes)
                )
                loss = loss.sum()
            else:
                raise ValueError(f"Invalid box loss type '{self.bbox_reg_loss_type}'")

            # Normalize by number of foreground anchors
            losses.append(loss / max(1, num_foreground))

        # Average loss across all images in batch
        return torch.stack(losses).mean()

    def _inference(
        self,
        cls_logits: Tensor,
        bbox_regression: Tensor,
        anchors: List[Tensor],
        image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        """
        Perform inference and generate detections

        Args:
            cls_logits: Classification logits for all anchors
            bbox_regression: Box regression values for all anchors
            anchors: Anchor boxes
            image_sizes: Original image sizes

        Returns:
            List of dictionaries containing:
            - 'boxes': Detected boxes
            - 'scores': Confidence scores
            - 'labels': Class labels

        Example flow:
            1. Apply box regression to anchors
            2. Convert sigmoid scores to probabilities
            3. Filter by confidence threshold
            4. Apply NMS per class
            5. Take top-k detections
        """
        device = cls_logits.device
        num_classes = cls_logits.shape[-1]

        # Create empty list to store results for each image
        results = []

        # Process one image at a time
        for i, (
            cls_logits_per_image,
            bbox_regression_per_image,
            anchors_per_image,
            image_size,
        ) in enumerate(zip(cls_logits, bbox_regression, anchors, image_sizes)):
            # Apply sigmoid to classification logits to get probabilities
            scores_per_image = cls_logits_per_image.sigmoid()

            # Decode box regression offsets to get actual box coordinates
            boxes_per_image = self.box_coder.decode(
                bbox_regression_per_image, anchors_per_image
            )

            # Create dict to store results for this image
            result = {
                "boxes": torch.zeros((0, 4), device=device),
                "labels": torch.zeros(0, dtype=torch.int64, device=device),
                "scores": torch.zeros(0, device=device),
            }

            # Process each class separately (except background)
            for class_idx in range(num_classes):
                # Filter by score threshold
                scores_for_class = scores_per_image[:, class_idx]
                keep_idxs = scores_for_class > self.score_thresh
                scores_for_class = scores_for_class[keep_idxs]

                # Skip if no detections for this class
                if scores_for_class.numel() == 0:
                    continue

                # Get boxes for kept indices
                boxes_for_class = boxes_per_image[keep_idxs, :]

                # Apply NMS for this class
                keep_idxs = batched_nms(
                    boxes_for_class,
                    scores_for_class,
                    torch.full_like(scores_for_class, class_idx),
                    self.nms_thresh,
                )

                # Take top detections after NMS
                keep_idxs = keep_idxs[: self.detections_per_img]

                # Add detections for this class to results
                result["boxes"] = torch.cat(
                    (result["boxes"], boxes_for_class[keep_idxs]), dim=0
                )
                result["scores"] = torch.cat(
                    (result["scores"], scores_for_class[keep_idxs]), dim=0
                )
                result["labels"] = torch.cat(
                    (
                        result["labels"],
                        torch.full_like(keep_idxs, class_idx, dtype=torch.int64),
                    ),
                    dim=0,
                )

            # If we have more than max detections, keep only the highest scoring ones
            if len(result["boxes"]) > self.detections_per_img:
                _, sorted_idxs = torch.sort(result["scores"], descending=True)
                sorted_idxs = sorted_idxs[: self.detections_per_img]
                result["boxes"] = result["boxes"][sorted_idxs]
                result["scores"] = result["scores"][sorted_idxs]
                result["labels"] = result["labels"][sorted_idxs]

            # Clip boxes to image size
            h, w = image_size
            result["boxes"] = box_ops.clip_boxes_to_image(result["boxes"], (h, w))

            results.append(result)

        return results

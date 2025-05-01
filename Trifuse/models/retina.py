from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

# import torchvision.models.detection._utils as det_utils
from torchvision.models.detection._utils import BoxCoder, Matcher, _topk_min
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.retinanet import (
    RetinaNetClassificationHead,
    RetinaNetRegressionHead,
)
from torchvision.ops import boxes as box_ops
from torchvision.ops import clip_boxes_to_image


class RetinaNet(nn.Module):
    """
    RetinaNet object detection model

    A one-stage detector using Feature Pyramid Network (FPN) for feature extraction
    and Focal Loss to address class imbalance problem.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        out_channels: int = 256,
        num_anchors: int = 9,
        anchor_generator: Optional[AnchorGenerator] = None,
        proposal_matcher: Optional[Matcher] = None,
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

        self.num_anchors = num_anchors

        self.cls_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )
        self.reg_head = RetinaNetRegressionHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
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
            proposal_matcher = Matcher(
                high_threshold=0.5, low_threshold=0.4, allow_low_quality_matches=True
            )

        self.proposal_matcher = proposal_matcher
        self.BETWEEN_THRESHOLDS = Matcher.BETWEEN_THRESHOLDS

        # Box coder for converting between formats
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

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
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ):
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
        """
        # Convert images to ImageList
        original_image_sizes = [img.shape[-2:] for img in images]
        image_list = ImageList(images, original_image_sizes)

        # Feed through RetinaNetHead to get logits
        head_outputs = {
            "cls_logits": self.cls_head(feature_maps),
            "bbox_regression": self.reg_head(feature_maps),
        }

        # Generate anchors for each feature map level
        anchors = self.anchor_generator(image_list, feature_maps)
        anchors = [
            clip_boxes_to_image(anchor, image_size)
            for anchor, image_size in zip(anchors, image_list.image_sizes)
        ]

        if self.training:
            return head_outputs, anchors
        else:
            return self._inference(head_outputs, anchors, image_list.image_sizes)

    def compute_loss(
        self,
        head_outputs: tuple[dict[str, Tensor], list[Tensor]],  # FIXME: wrong type hint
        targets: List[Dict[str, Tensor]],
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, Tensor]:
        # Match targets to anchors
        # Torchvision implementation. Source: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
        head_outputs, anchors = head_outputs
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
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

        # Compute classification loss
        cls_loss = self.cls_head.compute_loss(targets, head_outputs, matched_idxs)

        # Compute box regression loss
        box_loss = self.reg_head.compute_loss(
            targets, head_outputs, anchors, matched_idxs
        )

        return {
            "cls_loss": cls_loss,
            "bbox_reg_loss": box_loss,
            "tloss": cls_loss + self.box_loss_weight * box_loss,
        }

    def _inference(
        self, head_outputs, anchors, image_shapes
    ) -> List[Dict[str, Tensor]]:
        # Torchvision implementation. Source: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs],
                    anchors_per_level[anchor_idxs],
                )
                boxes_per_level = box_ops.clip_boxes_to_image(
                    boxes_per_level, image_shape
                )

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(
                image_boxes, image_scores, image_labels, self.nms_thresh
            )
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections

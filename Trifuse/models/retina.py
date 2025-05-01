from typing import Dict, List, Optional

import torch
import torch.nn as nn

# import torchvision.models.detection._utils as det_utils
import torchvision.models.detection._utils as det_utils
from torch import Tensor
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
        proposal_matcher: Optional[det_utils.Matcher] = None,
        box_loss_weight: float = 1.0,
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
        bbox_reg_loss_type: str = "smooth_l1",
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        topk_candidates: int = 1000,
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
            anchor_sizes = tuple(
                (s, int(s * 2 ** (1 / 3)), int(s * 2 ** (2 / 3)))
                for s in [32, 64, 128, 256]
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
        self.topk_candidates = topk_candidates
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

    # def _inference(
    #     self, head_outputs, anchors, image_shapes
    # ) -> List[Dict[str, Tensor]]:
    #     class_logits = head_outputs["cls_logits"]
    #     box_regression = head_outputs["bbox_regression"]
    #
    #     num_images = len(image_shapes)
    #     detections: List[Dict[str, Tensor]] = []
    #
    #     print("class_logits shape: ", class_logits.shape)
    #     print("box_regression shape: ", box_regression.shape)
    #
    #     def ensure_2d(x):
    #         return x.unsqueeze(0) if x.dim() == 1 else x
    #
    #     for img_idx in range(num_images):
    #         print(f"\n[DEBUG] === Image {img_idx} ===")
    #         box_regs = [br[img_idx] for br in box_regression]
    #         logits = [cl[img_idx] for cl in class_logits]
    #         img_anchors, img_shape = anchors[img_idx], image_shapes[img_idx]
    #
    #         print("box_regs shape: ", box_regs[0].shape)
    #         print("logits shape: ", logits[0].shape)
    #         print("img_anchors shape: ", img_anchors[0].shape)
    #
    #         for lvl, (br_lvl, logit_lvl, anch_lvl) in enumerate(
    #             zip(box_regs, logits, img_anchors)
    #         ):
    #             print(f"\n[DEBUG] -- Level {lvl} --")
    #             # raw shapes
    #             print("  br_lvl.shape:", br_lvl.shape)
    #             print("  logit_lvl.shape:", logit_lvl.shape)
    #             print("  anch_lvl.shape:", anch_lvl.shape)
    #
    #             num_classes = logit_lvl.shape[-1]
    #
    #             # scores
    #             scores = torch.sigmoid(logit_lvl).flatten()
    #             print("  flattened scores.shape:", scores.shape)
    #             keep = scores > self.score_thresh
    #             print("  keep mask.sum():", int(keep.sum().item()))
    #             scores = scores[keep]
    #             idxs = torch.where(keep)[0]
    #             print("  idxs.shape after threshold:", idxs.shape)
    #
    #             # top-K
    #             K = det_utils._topk_min(idxs, self.topk_candidates, 0)
    #             print("  top-K =", K)
    #             scores, topk_idxs = scores.topk(K)
    #             idxs = idxs[topk_idxs]
    #             print("  idxs.shape after topk:", idxs.shape)
    #
    #             # anchor + label indices
    #             anchor_idxs = idxs // num_classes
    #             labels = idxs % num_classes
    #             print("  anchor_idxs.shape:", anchor_idxs.shape)
    #             print("  labels.shape:", labels.shape)
    #             print("  labels.unique():", labels.unique().tolist())
    #
    #             # slice and ensure 2d
    #             sel_regs_raw = br_lvl[anchor_idxs]
    #             sel_anc_raw = anch_lvl[anchor_idxs]
    #             sel_regs = ensure_2d(sel_regs_raw)
    #             sel_anc = ensure_2d(sel_anc_raw)
    #             print(
    #                 "  sel_regs_raw.shape:",
    #                 sel_regs_raw.shape,
    #                 "→ ensure_2d:",
    #                 sel_regs.shape,
    #             )
    #             print(
    #                 "  sel_anc_raw.shape:",
    #                 sel_anc_raw.shape,
    #                 "→ ensure_2d:",
    #                 sel_anc.shape,
    #             )
    #
    #             # decode + clip
    #             boxes = self.box_coder.decode_single(sel_regs, sel_anc)
    #             print("  boxes.shape after decode:", boxes.shape)
    #             boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
    #             print("  boxes.shape after clip:", boxes.shape)
    #
    #         image_boxes = torch.cat(image_boxes, dim=0)
    #         image_scores = torch.cat(image_scores, dim=0)
    #         image_labels = torch.cat(image_labels, dim=0)
    #
    #         # non-maximum suppression
    #         keep = box_ops.batched_nms(
    #             image_boxes, image_scores, image_labels, self.nms_thresh
    #         )
    #         keep = keep[: self.detections_per_img]
    #
    #         detections.append(
    #             {
    #                 "boxes": image_boxes[keep],
    #                 "scores": image_scores[keep],
    #                 "labels": image_labels[keep],
    #             }
    #         )
    #
    #     return detections

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

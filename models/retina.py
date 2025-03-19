import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torchvision.ops import batched_nms, boxes as box_ops
import torchvision.models.detection._utils as det_utils
from torchvision.models.detection.retinanet import _sum, _box_loss


class Retina(nn.Module):
    """
    RetinaNet Head for object detection.

    This module consists of classification and regression subnets applied to feature maps
    from a Feature Pyramid Network (FPN). It supports optional feature map fusion to generate
    a single prediction instead of separate predictions for each level of the FPN.

    Args:
        num_classes (int): Number of object classes to detect.
        out_channels (int, optional): Number of output channels for the feature maps. Defaults to 192.
        fuse_fm (bool, optional): Whether to fuse feature maps before prediction. Defaults to True.
        num_fm (int, optional): Number of input feature maps. Defaults to 4.
        num_anchors (int, optional): Number of anchor boxes per feature map location. Defaults to 9.
        anchor_generator (AnchorGenerator, optional): AnchorGenerator object for anchors generation. Defaults to None.
        proposal_matcher (det_utils.Matcher, optional): det_utils.Matcher object to match anchor boxes. Defaults to None
        regression_loss_type (str, optional): Loss function for regression. Defaults to "smooth_l1".
    """

    def __init__(
        self,
        num_classes: int,
        out_channels: int = 256,
        fuse_fm: bool = True,
        num_fm: int = 4,
        num_anchors: int = 9,
        anchor_generator=None,
        proposal_matcher=None,
        regression_loss_type="smooth_l1",
    ):
        super().__init__()

        self.fuse_fm = fuse_fm
        self.num_fm = num_fm
        self.num_classes = num_classes
        if proposal_matcher is None:
            proposal_matcher = self._default_proposal_matcher(0.5, 0.3, True)
        self.proposal_matcher = proposal_matcher
        self.BETWEEN_THRESHHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        if anchor_generator:
            self.anchor_generator = anchor_generator
        else:
            self.anchor_generator = self._default_anchor_gen()

        self.reg_loss_type = regression_loss_type

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
            feature_maps = torch.cat(upsampled_maps, dim=1)
            feature_maps = [self.fuse(feature_maps)]

        cls_logits = []
        reg_logits = []

        for feature in feature_maps:
            cls_out = self.cls_logits(self.cls_subnet(feature))
            reg_out = self.bbox_reg(self.reg_subnet(feature))

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_out.shape
            cls_out = cls_out.view(N, -1, self.num_classes, H, W)
            cls_out = cls_out.permute(0, 3, 4, 1, 2)
            cls_out = cls_out.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = reg_out.shape
            reg_out = reg_out.view(N, -1, 4, H, W)
            reg_out = reg_out.permute(0, 3, 4, 1, 2)
            reg_out = reg_out.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            cls_logits.append(cls_out)
            reg_logits.append(reg_out)

        cls_logits = torch.cat(cls_logits, dim=1)
        reg_logits = torch.cat(reg_logits, dim=1)

        image_list = self._to_ImageList(images)
        anchors = self.anchor_generator(image_list, feature_maps)

        # If model training -> return loss
        if self.training:
            assert targets is not None, (
                "during training, targets of the images are required"
            )

            return self.compute_loss(targets, cls_logits, reg_logits, anchors)

        else:
            return self.postprocess(cls_logits, reg_logits, images, anchors)

    def _to_ImageList(self, images):
        """
        Converts a list of image tensors into an ImageList object.

        Args:
            images (list[torch.Tensor]): A list of image tensors, where each tensor has the shape (C, H, W), representing channels, height, and width.

        Returns:
            ImageList: An ImageList object containing the stacked images and their original sizes.

        Notes:
            - The `ImageList` class is typically used in object detection models to
              handle batched images of varying sizes.
            - `original_sizes` stores the height and width of each image before any processing.
        """
        original_sizes = [img.shape[-2:] for img in images]
        return ImageList(images, original_sizes)

    def _default_proposal_matcher(
        self, fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False
    ):
        """
        Creates a default proposal matcher for assigning ground truth boxes to anchors
        based on IoU (Intersection over Union) thresholds.

        Args:
            fg_iou_thresh (float): The IoU threshold above which an anchor is considered a foreground (positive) match.
            bg_iou_thresh (float): The IoU threshold below which an anchor is considered a background (negative) match.
            allow_low_quality_matches (bool, optional): If True, allows certain lower-quality matches
                to ensure each ground truth box has at least one matching anchor. Default is False.

        Returns:
            det_utils.Matcher: A matcher instance that assigns labels to anchors based on the IoU thresholds.
        """
        proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches,
        )
        return proposal_matcher

    def _default_anchor_gen(self):
        """
        Creates a default anchor generator with predefined sizes and aspect ratios for different
        feature pyramid levels.

        Returns:
            AnchorGenerator: An instance of `AnchorGenerator` configured with specific anchor sizes
            and aspect ratios.

        Notes:
            - The anchor sizes are defined for four feature map levels:
                - P1 (56x56): (16, 32, 64)
                - P2 (28x28): (32, 64, 128)
                - P3 (14x14): (64, 128, 192)
                - P4 (7x7): (128, 192, 224)
            - Each level has aspect ratios of (0.5, 1.0, 2.0), meaning:
                - 0.5: Tall and narrow anchors
                - 1.0: Square anchors
                - 2.0: Wide and short anchors
        """
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
        """
        Computes the classification and regression loss.

        Args:
            targets (list[dict]): Ground-truth annotations for each image in the batch.
                Each dictionary contains:
                    - "boxes" (torch.Tensor[N, 4]): Ground-truth bounding boxes.
                    - "labels" (torch.Tensor[N]): Class labels for the objects.
            cls_logits (torch.Tensor): Predicted classification logits of shape (batch_size, num_anchors, num_classes).
            bbox_reg (torch.Tensor): Predicted bounding box regression values of shape (batch_size, num_anchors, 4).
            anchors (torch.Tensor): Anchor boxes used for prediction, shape (num_anchors, 4).

        Returns:
            dict: A dictionary containing:
                - "cls_loss" (torch.Tensor): Classification loss.
                - "reg_loss" (torch.Tensor): Bounding box regression loss.
        """
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
    def _cls_loss(self, targets, cls_logits, matched_idxs, alpha=0.25, gamma=2):
        """
        Computes the classification loss using the sigmoid focal loss.

        Args:
            targets (list[dict]): List of target annotations for each image in the batch.
                Each dictionary contains:
                - "labels" (torch.Tensor[N]): Class labels for ground truth objects.
            cls_logits (list[torch.Tensor]): List of predicted classification logits for each image,
                with shape (num_anchors, num_classes).
            matched_idxs (list[torch.Tensor]): List of tensors containing indices of matched
                ground truth boxes for each anchor.
            alpha (float, optional): an argument for the focal loss function. Defaults to 0.25.
            gamma (int, optional): an argument for the focal loss function. Defaults to 2.

        Returns:
            Tensor: The computed classification loss averaged over all images.
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
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHHOLDS

            # compute the class classification loss
            losses.append(
                sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    alpha=alpha,
                    gamma=gamma,
                    reduction="sum",
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / len(targets)

    # PyTorch RetinaNetRegressionHead compute_loss()
    def _reg_loss(self, targets, bbox_reg, anchors, matched_idxs):
        """
        Computes the bounding box regression loss.

        Args:
            targets (list[dict]): List of target annotations for each image in the batch.
                Each dictionary contains:
                - "boxes" (torch.Tensor[N, 4]): Ground truth bounding boxes in (x1, y1, x2, y2) format.
            bbox_reg (list[torch.Tensor]): List of predicted bounding box deltas for each image,
                with shape (num_anchors, 4).
            anchors (list[torch.Tensor]): List of anchor boxes for each image, where each tensor
                has shape (num_anchors, 4).
            matched_idxs (list[torch.Tensor]): List of tensors containing indices of matched
                ground truth boxes for each anchor.

        Returns:
            torch.Tensor: The computed bounding box regression loss averaged over all images.
            - The final loss is normalized by the number of foreground samples and averaged over all images.
        """
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
                _box_loss(
                    self.reg_loss_type,
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
        """
        Post-processes the raw predictions from the model to obtain final detection results.

        Args:
            cls_logits (torch.Tensor): Classification logits of shape (batch_size, num_anchors, num_classes).
            reg_logits (torch.Tensor): Bounding box regression logits of shape (batch_size, num_anchors, 4).
            images (list): List of input images for reference (not directly used in processing).
            anchors (torch.Tensor): Anchor boxes used for predictions, shape (batch_size, num_anchors, 4).
            score_thresh (float, optional): Confidence threshold for filtering weak detections. Default is 0.5.
            nms_thresh (float, optional): Non-maximum suppression (NMS) threshold to remove redundant boxes. Default is 0.5.

        Returns:
            list of dict: A list of dictionaries, one per image, each containing:
                - "boxes" (torch.Tensor[N, 4]): Final detected bounding boxes after post-processing.
                - "scores" (torch.Tensor[N]): Confidence scores of the detections.
                - "labels" (torch.Tensor[N]): Predicted class labels for the detections.
        """
        results = []

        batch_size = cls_logits.shape[0]

        for img_idx in range(batch_size):
            # Extract per-image predictions
            cls_scores = cls_logits[img_idx]  # [112896, 80]
            reg_deltas = reg_logits[img_idx]  # [112896, 4]
            anchors_per_image = anchors[img_idx]

            # Convert logits to probabilities
            cls_probs = cls_scores.sigmoid()  # [112896, 80]

            boxes = self.box_coder.decode(
                reg_deltas, [anchors_per_image]
            )  # [112896, 1, 4]
            boxes = boxes.squeeze(1)  # -> [112896, 4]

            max_scores, labels = cls_probs.max(
                dim=1
            )  # max_scores: (112896,) labels: (112896,)

            keep = max_scores > score_thresh

            boxes, scores, labels = boxes[keep], max_scores[keep], labels[keep]

            # Apply NMS
            keep_idxs = batched_nms(boxes, scores, labels, nms_thresh)
            boxes, scores, labels = (
                boxes[keep_idxs],
                scores[keep_idxs],
                labels[keep_idxs],
            )

            results.append({"boxes": boxes, "scores": scores, "labels": labels})

        return results

    def match_anchors(self, targets, anchors):
        """
        Matches anchor boxes to ground truth boxes for each image in the batch.

        Args:
            targets (list[dict]): List of target annotations for each image in the batch.
                Each dictionary contains:
                - "boxes" (torch.Tensor[N, 4]): Ground truth bounding boxes in (x1, y1, x2, y2) format.
            anchors (list[torch.Tensor]): List of anchor boxes for each image, where each tensor
                has shape (num_anchors, 4).

        Returns:
            list[torch.Tensor]: A list of tensors, one per image, containing the indices of the matched
            ground truth boxes for each anchor.
            - If no ground truth boxes exist, a tensor of shape (num_anchors,) filled with -1 is returned.
            - Otherwise, the indices indicate which ground truth box each anchor is matched to.
        """
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

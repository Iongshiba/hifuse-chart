import math
import copy
import torch

from torch import nn
from torchvision.ops.misc import MLP
from scipy.optimize import linear_sum_assignment
from utils.misc import get_clones, box_cxcywh_to_xyxy, generalized_iou
import itertools

import torch.nn.functional as F


class DETR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_queries: int = 100,
        hidden_dim: int = 256,
        feedforward_dim: int = 2048,
        num_heads: int = 8,
        encoder_num: int = 6,
        decoder_num: int = 6,
        temperature: int = 10000,
        aux_loss: bool = True,
        he_gain: float = 1.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.aux_loss = aux_loss

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # 4 is the number of coords in a bounding box
        self.bbox_embed = MLP(
            in_channels=hidden_dim,
            hidden_channels=[hidden_dim, hidden_dim, 4],
            activation_layer=nn.LeakyReLU,
        )
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        ###### Transformer Setting ######

        self.encoder_layer = TransformerEncoderLayer(
            hidden_dim, feedforward_dim, num_heads
        )
        self.decoder_layer = TransformerDecoderLayer(
            hidden_dim, feedforward_dim, num_heads
        )

        self.encoder = TransformerEncoder(self.encoder_layer, encoder_num)
        self.decoder = TransformerDecoder(
            self.decoder_layer, decoder_num, nn.LayerNorm(hidden_dim), aux_loss
        )

        self._init_weights_scaled_he(he_gain)

        ###### Loss Setting ######

    def _pos_embed(self, x):
        if len(x.shape) == 4:
            B, _, H, W = x.shape
        elif len(x.shape) == 3:
            B, W, _ = x.shape
            H = 1

        # divide because of 2d positional embedding
        embed_dim = self.hidden_dim // 2

        # dim_t = [t^(2*0/hidden_dim), t^(2*1/hidden_dim), t^(2*2/hidden_dim), t^(2*3/hidden_dim)]
        dim_t = torch.arange(0, embed_dim, dtype=torch.float32, device=x.device)
        dim_t = torch.pow(
            self.temperature,
            2 * (dim_t // 2) / embed_dim,
        )

        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=x.device)
        x_pos, y_pos = torch.meshgrid(y_embed, x_embed, indexing="ij")
        x_pos = x_pos.repeat(B, 1, 1)
        y_pos = y_pos.repeat(B, 1, 1)

        x_pos = x_pos[:, :, :, None] / dim_t
        y_pos = y_pos[:, :, :, None] / dim_t
        x_pos = torch.stack(
            (x_pos[:, :, :, 0::2].sin(), x_pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        y_pos = torch.stack(
            (y_pos[:, :, :, 0::2].sin(), y_pos[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        # print(x_pos)
        # print(y_pos)

        pos = torch.cat((x_pos, y_pos), dim=3).permute(0, 3, 1, 2)

        if len(x.shape) == 3:
            pos = torch.squeeze(pos, dim=2)

        # print(pos.shape)

        return pos

    def _get_aux_loss(self, out_class, out_bbox):
        return [
            {
                f"pred_logits": c,
                f"pred_boxes": b,
            }
            for i, (c, b) in enumerate(zip(out_class[:-1], out_bbox[:-1]))
        ]

    def _init_weights_scaled_he(self, custom_gain_factor=2.0):
        """
        Applies He Normal initialization scaled by a custom factor.
        """
        for m in self.bbox_embed:
            if isinstance(m, nn.Linear):
                if hasattr(m, "weight") and m.weight is not None:
                    # Calculate fan_in (number of input units)
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)

                    # Calculate standard He gain for ReLU
                    standard_he_gain = nn.init.calculate_gain(
                        "leaky_relu"
                    )  # This is sqrt(2)

                    # Apply the custom scaling factor
                    effective_gain = standard_he_gain * custom_gain_factor

                    # Calculate the desired standard deviation
                    # stddev = effective_gain * sqrt(1 / fan_in) is equivalent to
                    # stddev = sqrt( (effective_gain**2) / fan_in )
                    # Note: Original He is sqrt(2/fan_in) which is gain*sqrt(1/fan_in)
                    # So scaled version is custom_gain_factor * sqrt(2/fan_in)
                    # OR effective_gain / sqrt(fan_in)

                    std = effective_gain / math.sqrt(fan_in)

                    # Initialize weights using basic normal distribution
                    nn.init.normal_(m.weight, mean=0.0, std=std)
                    print(
                        f"Initialized layer with fan_in={fan_in}, custom_gain={effective_gain:.2f}, std={std:.4f}"
                    )

                # Initialize bias to zero (common practice)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        B, C, H, W = x.shape

        # [B, in_c * 4, H, W] -> [B, hidden_dim, H, W]
        x = self.input_proj(x)

        # 2d positional embedding
        # [B, hidden_dim, H, W]
        pos_embed = self._pos_embed(x)

        # flattening and prepare for MultiheadAttention
        # [B, C, H, W] -> [B, C, H * W] -> [H * W, B, C]
        x = x.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        memory = self.encoder(x, pos_embed)
        target = torch.zeros_like(query_embed)
        output = self.decoder(target, memory, pos_embed, query_embed)

        # [intermediate, num_queries, batch_size, hidden_dim]
        # print(output.shape)

        # [intermediate, batch_size, num_queries, num_classes + 1]
        # [intermediate, batch_size, num_queries, 4]
        out_class = self.class_embed(output).permute(0, 2, 1, 3)
        out_bbox = self.bbox_embed(output)
        out_bbox = out_bbox.sigmoid().permute(0, 2, 1, 3)

        # targets = [
        #     {
        #         "labels": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        #         "boxes": torch.tensor(
        #             [
        #                 [20.2, 17.6, 44.7, 24.8],
        #                 [20.0, 44.7, 43.8, 24.3],
        #                 [120.0, 16.7, 41.2, 25.4],
        #                 [48.2, 17.7, 44.7, 25.8],
        #                 [48.1, 44.4, 45.0, 26.2],
        #                 [156.7, 16.7, 38.7, 25.7],
        #                 [120.0, 29.1, 41.2, 25.4],
        #                 [156.7, 28.7, 38.7, 24.8],
        #                 [119.4, 53.0, 45.7, 25.9],
        #                 [157.7, 53.2, 41.8, 27.1],
        #                 [119.3, 67.6, 44.3, 27.4],
        #                 [157.6, 67.4, 45.7, 27.0],
        #             ]
        #         ),
        #     }
        # ]
        outputs = {
            "pred_logits": out_class[-1],
            "pred_boxes": out_bbox[-1],
        }

        if self.aux_loss:
            outputs["aux_outputs"] = self._get_aux_loss(out_class, out_bbox)

        # criterion = SetCriterion(1)
        # loss = criterion(outputs, targets)

        return outputs


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, x, pos_embed):
        output = x

        for layer in self.layers:
            output = layer(output, pos_embed)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, aux_loss=False):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.norm = norm
        self.aux_loss = aux_loss

    def forward(self, x, memory, pos_embed, query_embed):
        output = x
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, pos_embed, query_embed)
            if self.aux_loss:
                intermediate.append(self.norm(output))

        if self.norm:
            output = self.norm(output)
            if self.aux_loss:
                intermediate.pop()
                intermediate.append(output)

        if self.aux_loss:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, out_channels, ffn_dim, num_heads, dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout

        self.attn = nn.MultiheadAttention(out_channels, num_heads, dropout)
        self.norm1 = nn.LayerNorm(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = MLP(
            out_channels,
            [ffn_dim, out_channels],
            dropout=dropout,
            activation_layer=nn.LeakyReLU,
        )
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, pos_embed):
        # positional memoryedding at each encoding layer
        q = k = x + pos_embed

        # self-attention
        x_attn, _ = self.attn(q, k, x)
        x = x + self.dropout1(x_attn)
        x = self.norm1(x)

        # feedforward
        x_ffn = self.ffn(x)
        x = x + self.dropout2(x_ffn)
        x = self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, out_channels, ffn_dim, num_heads, dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout

        self.self_attn = nn.MultiheadAttention(out_channels, num_heads, dropout)
        self.norm1 = nn.LayerNorm(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(out_channels, num_heads, dropout)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = MLP(
            out_channels,
            [ffn_dim, out_channels],
            dropout=dropout,
            activation_layer=nn.LeakyReLU,
        )
        self.norm3 = nn.LayerNorm(out_channels)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, pos_embed, query_embed):
        # positional embedding at each decoding layer
        q_self = k_self = x + query_embed

        # self-attention
        self_attn, _ = self.self_attn(q_self, k_self, x)
        x = x + self.dropout1(self_attn)
        x = self.norm1(x)

        # cross-attention with box query and positional embedding
        q_cross = x + query_embed
        k_cross = memory + pos_embed
        cross_attn, _ = self.cross_attn(q_cross, k_cross, memory)
        x = x + self.dropout2(cross_attn)
        x = self.norm2(x)

        x_ffn = self.ffn(x)
        x = x + self.dropout3(x_ffn)
        x = self.norm3(x)

        return x


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes,
        weight_dict={"ce_coeff": 1.0, "l1_coeff": 5.0, "giou_coeff": 2.0},
        non_object_coeff=0.1,
        non_object_index=0,
    ):
        super().__init__()
        self.matcher = HungarianMatcher()
        self.non_obj_idx = non_object_index
        self.num_classes = num_classes
        cross_entropy_weight = torch.ones(num_classes + 1)
        cross_entropy_weight[non_object_index] = non_object_coeff
        self.weight_dict = weight_dict
        self.register_buffer("cross_entropy_weight", cross_entropy_weight)

    def loss_labels(self, outputs, targets, indices):
        # tuple of index of all src with one-to-one map to target: sum(labels_of_images_in_batch)
        idx = self._get_src_permutation_idx(indices)
        # [batch_size, num_queries, num_classes + 1]
        src_logits = outputs["pred_logits"]
        # total labels in a batch
        tgt_label = torch.cat([t["labels"][i] for t, (_, i) in zip(targets, indices)])
        # create [batch_size, num_queries]
        tgt_classes = torch.full(
            src_logits.shape[:2],
            self.non_obj_idx,
            dtype=torch.int64,
            device=src_logits.device,
        )
        tgt_classes[idx] = tgt_label

        # cross entropy between (batch_size, num_classes + 1, num_queries) and (batch_size, num_queries), where:
        # - First one contains the predicted logits for all classes.
        # - Second one contains classes
        # For example one batch, and 2 classes + 1 non object class:
        # [0.23, 0.15, 0.231, 0.25, ...] len = 100
        # [0.23, 0.15, 0.231, 0.25, ...] len = 100
        # [0.23, 0.15, 0.231, 0.25, ...] len = 100
        # [0,    1,    2,     0,    ...] len = 100
        # cross entropy is computed at each column.
        ce_loss = F.cross_entropy(
            src_logits.transpose(1, 2),
            tgt_classes,
            self.cross_entropy_weight,
        )

        loss = {"ce_loss": ce_loss * self.weight_dict["ce_coeff"]}

        return loss

    def loss_boxes(self, outputs, targets, indices):
        # Non object mask: ignore computing loss for non object boundigng box
        mask = self._get_object_mask(targets, indices)

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx][mask]
        tgt_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)])[
            mask
        ]

        # For normalization across each batch
        num_boxes = len(tgt_boxes) + 1e-6
        num_boxes = torch.as_tensor(
            num_boxes, dtype=torch.float32, device=src_boxes.device
        )

        l1_loss = F.l1_loss(src_boxes, tgt_boxes, reduction="none")
        l1_loss = l1_loss.sum() / num_boxes

        giou_loss = 1 - torch.diag(
            generalized_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)
            )
        )
        giou_loss = giou_loss.sum() / num_boxes

        loss = {
            "l1_loss": l1_loss * self.weight_dict["l1_coeff"],
            "giou_loss": giou_loss * self.weight_dict["giou_coeff"],
        }

        return loss

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        # number of labels in an image
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        src_idx = torch.cat([tgt for (_, tgt) in indices])

        return batch_idx, src_idx

    def _get_object_mask(self, targets, indices):
        tgt_label = torch.cat([t["labels"][i] for t, (_, i) in zip(targets, indices)])

        return tgt_label != self.non_obj_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)

        loss = {}
        loss.update(self.loss_boxes(outputs, targets, indices))
        loss.update(self.loss_labels(outputs, targets, indices))

        # handle auxiliary losses
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)

                aux_loss = {}
                aux_loss.update(self.loss_boxes(aux_outputs, targets, indices))
                aux_loss.update(self.loss_labels(aux_outputs, targets, indices))
                aux_loss = {k + f"_{i}": v for k, v in aux_loss.items()}

                loss.update(aux_loss)

        # loss = {
        #     "l1_loss": (1),
        #     "giou_loss": (1),
        #     "ce_loss": (1),
        # }

        return loss


class HungarianMatcher(nn.Module):
    def __init__(
        self, class_weight: float = 1, bbox_weight: float = 1, l1_weight: float = 1
    ):
        super().__init__()
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.l1_weight = l1_weight

    def forward(self, outputs, targets):
        B, N, _ = outputs["pred_logits"].shape
        # outputs: This is a dict that contains at least these entries:
        #          "pred_logits": Tensor of dim [batch_size, num_queries, num_classes + 1] with the classification logits
        #          "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

        # targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
        #         "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
        #                 objects in the target) containing the class labels
        #         "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        out_class = outputs["pred_logits"]
        out_bbox = outputs["pred_boxes"]
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # num_classes = num_boxes in targets
        # [batch_size, num_queries, num_classes + 1] -> [batch_size * num_queries, num_classes + 1]
        # [batch_size, num_queries, 4] -> [batch_size * num_queries, 4]
        out_prob = out_class.flatten(0, 1).softmax(-1)
        out_bbox = out_bbox.flatten(0, 1)

        # In the matching cost, we use probabilities instead of log-probabilities.
        # This makes the class prediction term commensurable (same standard) to box loss, and we observed better empirical performances.
        class_cost = -out_prob[:, tgt_ids]
        bbox_cost = -generalized_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )
        l1_cost = torch.cdist(out_bbox, tgt_bbox, p=1)

        # [batch_size * num_queries, total tgt_bbox in the batch]: compute and only take the cost along the diagonal of the matrix
        cost = (
            self.class_weight * class_cost
            + self.bbox_weight * bbox_cost
            + self.l1_weight * l1_cost
        )
        # The [batch_size * num_queries, total tgt_bbox in the batch] is revert to original batch,
        # with each element in the batch responsible for each image in the targets
        cost = cost.view(B, N, -1).detach().cpu()

        # len(tgt_bbox_per_img) = batch_sizae
        tgt_bbox_per_img = [len(v["labels"]) for v in targets]
        # [batch_size, tuples each contains pair (output_bbox, tgt_bbox) in an image]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(cost.split(tgt_bbox_per_img, -1))
        ]

        # convert to torch.tensor before returning
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


if __name__ == "__main__":
    x = torch.ones((16, 7, 3))
    detr = DETR(
        96,
        1,
        100,
        8,
    )

    print(detr._pos_embed(x))

import copy
import torch

from torch import nn
from torchvision.ops.misc import MLP
from scipy.optimize import linear_sum_assignment
from utils.misc import get_clones, box_cxcywh_to_xyxy, generalized_iou

import torch.nn.functional as F


class DETR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_queries: int,
        hidden_dim: int,
        feedforward_dim: int = 2048,
        num_heads: int = 8,
        encoder_num: int = 6,
        decoder_num: int = 6,
        temperature: int = 10000,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # 4 is the number of coords in a bounding box
        self.bbox_embed = MLP(
            in_channels=hidden_dim,
            hidden_channels=[hidden_dim, hidden_dim, 4],
            activation_layer=nn.ReLU,
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
            self.decoder_layer, decoder_num, nn.LayerNorm(hidden_dim)
        )

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
        x_pos, y_pos = torch.meshgrid(y_embed, x_embed)
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

    def forward(self, x):
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

        out_class = self.class_embed(output)
        out_bbox = self.bbox_embed(output)

        import cv2 as cv

        targets = [
            {
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[0.471, 0.564, 0.130, 0.214]]),
            }
        ]

        criterion = HungarianMatcher()

        return criterion({"pred_logits": out_class, "pred_boxes": out_bbox}, targets)
        # return output


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
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.norm = norm

    def forward(self, x, memory, pos_embed, query_embed):
        output = x

        for layer in self.layers:
            output = layer(output, memory, pos_embed, query_embed)

        if self.norm:
            output = self.norm(output)

        return output


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
            activation_layer=nn.ReLU,
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
            activation_layer=nn.ReLU,
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
    def __init__():
        super().__init__()

    def loss_boxes(self, outputs, targets, indices):
        pass

    def forward(self):
        pass


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
        #          "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
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
        # [batch_size, num_queries, num_classes] -> [batch_size * num_queries, num_classes]
        # [batch_size, num_queries, 4] -> [batch_size * num_queries, 4]
        out_prob = out_class.flatten(0, 1).softmax(-1)
        out_bbox = out_bbox.flatten(0, 1)

        # In the matching cost, we use probabilities instead of log-probabilities.
        # This makes the class prediction term commensurable (same standard) to box loss, and we observed better empirical performances.
        class_cost = -out_prob[:, tgt_ids]
        bbox_cost = generalized_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )
        l1_cost = torch.cdist(out_bbox, tgt_bbox, p=1)

        # [batch_size * num_queries, total tgt_bbox in the batch]
        cost = (
            self.class_weight * class_cost
            + self.bbox_weight * bbox_cost
            + self.l1_weight * l1_cost
        )
        cost = cost.view(B, N, -1).cpu()

        # len(tgt_bbox_per_img) = batch_sizae
        tgt_bbox_per_img = [len(v["labels"]) for v in targets]
        # [batch_size, tuples each contains pair (output_bbox, tgt_bbox) in an image]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(cost.split(tgt_bbox_per_img, -1))
        ]

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


class SetCriterion(nn.Module):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    x = torch.ones((16, 7, 3))
    detr = DETR(
        96,
        1,
        100,
        8,
    )

    print(detr._pos_embed(x))

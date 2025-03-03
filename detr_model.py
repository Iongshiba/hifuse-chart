import torch

from torch import nn
from torchvision.ops.misc import MLP

import torch.nn.functional as F


class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, hidden_dim):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # 4 is the number of coords in a bounding box
        self.bbox_embed = MLP(
            in_channels=hidden_dim,
            hidden_channels=[hidden_dim, hidden_dim, 4],
            activation_layer=nn.ReLU,
        )
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, 1)

    def forward(self, x):
        pass

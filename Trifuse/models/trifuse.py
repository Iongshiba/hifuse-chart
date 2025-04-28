# Integrate between Trifuse backbone and detection head

import torch.nn as nn


class TriFuse(nn.Module):
    def __init__(self, backbone, head):
        super(TriFuse, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        pass

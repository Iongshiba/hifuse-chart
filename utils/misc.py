import copy
import torch
import numpy as np
import torch.nn as nn
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def generalized_iou(boxes1, boxes2):
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = default_iou(boxes1, boxes2)

    tl = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    br = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (br - tl).clamp(min=0)

    area = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (area - union) / area


def default_iou(boxes1, boxes2):
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # [num, 4] -> matrix of iou pair-wise between boxes
    tl = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    br = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # [num_box1, num_box2, 2]
    wh = (br - tl).clamp(min=0)

    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def get_clones(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])

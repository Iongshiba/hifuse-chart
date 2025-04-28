import os
import copy
import torch
import pickle
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from torchvision.ops.boxes import box_area

import torch.nn as nn
import matplotlib.pyplot as plt


def write_pickle(list_info: list, file_name: str):
    with open(file_name, "wb") as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, "rb") as f:
        info_list = pickle.load(f)
        return info_list


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]

    return torch.stack(b, dim=-1)


def box_cxcywh_to_xywh(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), w, h]

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

    return giou


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


def plot_bboxes_batch(
    images, predicted_bboxes_batch, ground_truth_bboxes_batch, batch_size
):
    grid_size = int(
        np.ceil(np.sqrt(batch_size))
    )  # Determine the grid dimensions (square-like)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15), squeeze=False)
    axes = axes.flatten()

    for i in range(batch_size):
        image = images[i]
        predicted_bboxes = predicted_bboxes_batch[i]
        ground_truth_bboxes = ground_truth_bboxes_batch[i]

        ax = axes[i]
        ax.imshow(image)

        for gt_bbox in ground_truth_bboxes:
            x, y, w, h = gt_bbox
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor="g", facecolor="none")
            ax.add_patch(rect)

        for pred_bbox in predicted_bboxes:
            x, y, w, h = pred_bbox
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)

        ax.axis("off")  # Turn off axis for better visualization

    for i in range(batch_size, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return fig


def check_model_memory(model, criterion, dataloader):
    torch.cuda.empty_cache()
    initial_mem = torch.cuda.memory_allocated()

    img, anns, _ = next(iter(dataloader))
    anns = [{k: v.cuda() for k, v in t.items()} for t in anns]

    outputs = model(img.cuda())

    peak_mem_forward = torch.cuda.max_memory_allocated()

    losses = criterion(outputs, anns)
    loss = sum(losses.values())
    loss.backward()

    peak_mem_total = torch.cuda.max_memory_reserved()

    print(f"Initial Memory: {initial_mem / 1e9:.2f} GB")
    print(f"Model + Activations: ~{peak_mem_forward / 1e9:.2f} GB")
    print(
        f"Model + Activations + Gradients + Optimizer: ~{peak_mem_total / 1e9:.2f} GB"
    )
    print(
        f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )

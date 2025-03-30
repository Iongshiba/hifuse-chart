import os
import copy
import json
import glob
import torch
import pickle
import numpy as np
import torch.nn as nn

from PIL import Image
from torchvision.ops.boxes import box_area
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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


def confusion_matrix(x, y, num_classes):
    indices = num_classes * y + x
    matrix = torch.bincount(indices, minlength=num_classes**2).view(
        num_classes, num_classes
    )
    return matrix


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = "./class_indices.json"
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, "r")
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype("uint8"))
        plt.show()


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

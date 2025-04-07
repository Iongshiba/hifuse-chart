import json
import math
import copy
import torch

import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from models.trifuse import TriFuse
from models.detr import SetCriterion
from utils.data import (
    read_data_detection_yolo,
    read_data_detection_coco,
    make_coco_transforms,
    DetectionTransform,
)
from data.doclaynet import YOLODataset, COCODataset


def TriFuse_Tiny(num_classes: int, head: str = "detr"):

    model = TriFuse(
        depths=(2, 2, 2, 2),
        conv_depths=(2, 2, 2, 2),
        num_classes=num_classes,
        head=head,
    )
    return model


def TriFuse_Small(num_classes: int, head: str = "detr"):
    model = TriFuse(
        depths=(2, 2, 6, 2),
        conv_depths=(2, 2, 6, 2),
        num_classes=num_classes,
        head=head,
    )
    return model


def TriFuse_Base(num_classes: int, head: str = "detr"):
    model = TriFuse(
        depths=(2, 2, 18, 2),
        conv_depths=(2, 2, 18, 2),
        num_classes=num_classes,
        head=head,
    )
    return model


def create_dataset(args):
    train_dataset = None
    val_dataset = None

    if args.data == "yolo":
        train_images_path, train_images_label, val_images_path, val_images_label = (
            read_data_detection_yolo(args.root_data_path)
        )
        train_dataset = YOLODataset(
            images_path=train_images_path,
            labels_path=train_images_label,
            transform=DetectionTransform(
                make_coco_transforms("train", args.image_size)
            ),
        )
        val_dataset = YOLODataset(
            images_path=val_images_path,
            labels_path=val_images_label,
            transform=DetectionTransform(make_coco_transforms("val", args.image_size)),
        )
    elif args.data == "coco":
        train_images_dir, train_label_path, val_images_dir, val_label_path = (
            read_data_detection_coco(args.root_data_path)
        )
        train_dataset = COCODataset(
            image_dir=train_images_dir,
            label_path=train_label_path,
            transform=DetectionTransform(
                make_coco_transforms("train", args.image_size)
            ),
        )
        val_dataset = COCODataset(
            image_dir=val_images_dir,
            label_path=val_label_path,
            transform=DetectionTransform(make_coco_transforms("val", args.image_size)),
        )

    return train_dataset, val_dataset


def create_criterion(num_classees: int, head: str = "detr"):
    criterion = None
    if head == "detr":
        criterion = SetCriterion(num_classes=num_classees)
    elif head == "retina":
        criterion = None
    return criterion


def create_lr_scheduler(
    optimizer,
    num_step: int,
    epochs: int,
    warmup=True,
    warmup_epochs=1,
    warmup_factor=1e-3,
    end_factor=1e-2,
):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = x - warmup_epochs * num_step
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (
                1 - end_factor
            ) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(
    model: torch.nn.Module, weight_decay: float = 1e-5, learning_rate: float = 1e-4
):
    parameter_group_vars = {
        "backbone_decay": {
            "params": [],
            "weight_decay": weight_decay,
            "lr": learning_rate * 0.1,  # Lower LR for backbone
        },
        "backbone_no_decay": {
            "params": [],
            "weight_decay": 0.0,
            "lr": learning_rate * 0.1,
        },
        "head_decay": {
            "params": [],
            "weight_decay": weight_decay,
            "lr": learning_rate,  # Higher LR for head
        },
        "head_no_decay": {"params": [], "weight_decay": 0.0, "lr": learning_rate},
    }

    parameter_group_names = copy.deepcopy(parameter_group_vars)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters

        if len(param.shape) == 1 or name.endswith(".bias"):
            decay_status = "no_decay"
        else:
            decay_status = "decay"

        if "head" in name.lower():
            section = "head"
        else:
            section = "backbone"

        group_name = f"{section}_{decay_status}"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # Save group names (optional)
    # with open("parameter.json", "w") as file:
    #     json.dump(parameter_group_names, file, indent=2)

    return list(parameter_group_vars.values())

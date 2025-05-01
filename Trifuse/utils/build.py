import copy
import math
import os
from typing import Union

import torch
from torch.utils.data import DataLoader, DistributedSampler

from Trifuse.data.dataset import COCODataset, YOLOCOCODataset, YOLODataset
from Trifuse.models.backbone import TriFuseBackbone
from Trifuse.models.detr import SetCriterion, DETR
from Trifuse.models.retina import RetinaNet
from utils.data import (
    DetectionTransform,
    ImageOnlyTransform,
    make_coco_transforms,
    read_data_detection_coco,
    read_data_detection_yolo,
)


def build_model(num_classes: int, head_type: str, variant: str = "tiny"):
    assert (
        isinstance(num_classes, int) and num_classes > 0
    ), "num_classes should be a positive integer"
    assert variant in [
        "tiny",
        "small",
        "base",
    ], f"Unknown variant: {variant}, expected 'tiny', 'small', or 'base'"
    assert head_type in [
        "detr",
        "retina",
    ], f"Unknown head: {head_type}, expected 'detr' or 'retina'"

    assert (
        isinstance(num_classes, int) and num_classes > 0
    ), "num_classes should be a positive integer"
    assert variant in [
        "tiny",
        "small",
        "base",
    ], f"Unknown variant: {variant}, expected 'tiny', 'small', or 'base'"

    if variant == "tiny":
        depths = (2, 2, 2, 2)
        conv_depths = (2, 2, 2, 2)
    elif variant == "small":
        depths = (2, 2, 6, 2)
        conv_depths = (2, 2, 6, 2)
    else:  # base
        depths = (2, 2, 18, 2)
        conv_depths = (2, 2, 18, 2)

    backbone = TriFuseBackbone(
        depths=depths,
        conv_depths=conv_depths,
        num_classes=num_classes,
    )
    if head_type == "retina":
        head = RetinaNet(
            num_classes=num_classes,
        )
    else:
        head = DETR(
            in_channels=32 * 8 * 4,
            num_classes=num_classes,
        )

    return backbone, head


def build_dataset(
    data_type: str,
    image_size: int,
    root_path: str,
    disable_bbox_transform: bool = False,
):
    assert root_path is not None, f"Dataset root path {root_path} does not exist"

    train_dataset = None
    val_dataset = None

    if data_type == "yolo":
        train_images_path, train_images_label, val_images_path, val_images_label = (
            read_data_detection_yolo(root_path)
        )
        train_dataset = YOLODataset(
            images_path=train_images_path,
            labels_path=train_images_label,
            transform=DetectionTransform(make_coco_transforms("train", image_size)),
        )
        val_dataset = YOLODataset(
            images_path=val_images_path,
            labels_path=val_images_label,
            transform=DetectionTransform(make_coco_transforms("val", image_size)),
        )
    elif data_type == "coco":
        train_images_dir, train_label_path, val_images_dir, val_label_path = (
            read_data_detection_coco(root_path)
        )
        if disable_bbox_transform:
            train_dataset = COCODataset(
                image_dir=train_images_dir,
                label_path=train_label_path,
                transform=ImageOnlyTransform(make_coco_transforms("train", image_size)),
            )
            val_dataset = COCODataset(
                image_dir=val_images_dir,
                label_path=val_label_path,
                transform=ImageOnlyTransform(make_coco_transforms("val", image_size)),
            )
        else:
            train_dataset = COCODataset(
                image_dir=train_images_dir,
                label_path=train_label_path,
                transform=DetectionTransform(make_coco_transforms("train", image_size)),
            )
            val_dataset = COCODataset(
                image_dir=val_images_dir,
                label_path=val_label_path,
                transform=DetectionTransform(make_coco_transforms("val", image_size)),
            )

    return train_dataset, val_dataset


def build_dataloader(
    dataset: Union[COCODataset, YOLODataset, YOLOCOCODataset],
    shuffle: bool,
    batch_size: int,
    rank: int,
    num_workers: int = 8,
    seed: int = 42,
    pin_memory: bool = True,
):
    sampler = None if rank == -1 else DistributedSampler(dataset, shuffle=shuffle)
    num_workers = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, num_workers]
    )  # number of workers
    generator = torch.Generator()
    generator.manual_seed(seed + rank)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=pin_memory,
        collate_fn=dataset.collate_fn,
        generator=generator,
    )


def build_criterion(num_classes: int, head: str = "detr"):
    criterion = None
    if head == "detr":
        criterion = SetCriterion(num_classes=num_classes)
    elif head == "retina":
        criterion = None
    return criterion


def build_lr_scheduler(
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

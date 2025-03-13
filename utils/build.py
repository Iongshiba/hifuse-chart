import math
import torch


import torch.nn as nn
import torch.nn.functional as F

from models.trifuse import TriFuse
from models.detr import SetCriterion


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


def create_criterion(num_classees: int, head: str = "detr", **kwarg):
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


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {
        "decay": {"params": [], "weight_decay": weight_decay},
        "no_decay": {"params": [], "weight_decay": 0.0},
    }

    parameter_group_names = {
        "decay": {"params": [], "weight_decay": weight_decay},
        "no_decay": {"params": [], "weight_decay": 0.0},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

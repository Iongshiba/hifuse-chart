from argparse import Namespace
from typing import List, Union
import wandb
import os
import yaml
from pathlib import Path
from torch import nn
from Trifuse.utils import DEFAULT_CONFIG_FILE, RANK
from Trifuse.utils.build import build_model
from Trifuse.utils.trainer import TriFuseTrainer


class TriFuseDetector(nn.Module):
    def __init__(self, num_classes: int, head_type: str, variant: str = "tiny"):
        super(TriFuseDetector, self).__init__()
        self.backbone, self.head = build_model(num_classes, head_type, variant)

    def forward(self, x):
        return self.head(self.backbone(x))

    def compute_loss(self, outputs, targets, criterion):
        return self.head.compute_loss(outputs, targets, criterion)


class TriFuse:
    def __init__(
        self,
        num_classes: int,
        head: str,
        variant: str = "tiny",
    ):
        self.model = TriFuseDetector(num_classes, head, variant)
        self.args = None
        self.logger = None
        self.trainer = None

    def train(
        self,
        config_path: Union[str, Path] = DEFAULT_CONFIG_FILE,
        **overrides,
    ):
        # Load config file
        print("[INFO] Loading config file...")
        if "config_path" in overrides:
            config_path = overrides.pop("config_path")
        self.args = self._load_cfg(config_path, overrides)

        # Initialize Wandb
        print("[INFO] Initializing Wandb...")
        self._wandb_init(self.args.enable_logger)

        # Initialize trainer
        print("[INFO] Initializing trainer...")
        self.trainer = TriFuseTrainer(self.args)

        # Start training
        print("[INFO] Starting training...")
        self.trainer.train()

        # Done training
        print("[INFO] Training done.")

    def _load_cfg(self, cfg_path: Union[str, Path], overrides: dict) -> Namespace:
        if isinstance(cfg_path, (str, Path)):
            with open(cfg_path, errors="ignore", encoding="utf-8") as f:
                data = f.read()
                cfg = yaml.safe_load(data) or {}
        for key, value in overrides.items():
            cfg[key] = value

        cfg = Namespace(**cfg)
        return cfg

    def _wandb_init(self, enable_logger: bool):
        if RANK == 0 and enable_logger:
            wandb.login(key=os.environ["WANDB_API_KEY"])
            self.logger = wandb.init(project="trifuse", config=vars(self.args))
            self.logger.define_metric("eval/precision", summary="max")
            self.logger.define_metric("eval/recall", summary="max")
            self.logger.define_metric("eval/mAP50", summary="max")
            self.logger.define_metric("eval/mAP5095", summary="max")

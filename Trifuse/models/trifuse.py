import os
from pathlib import Path
from typing import Union

from torch import nn

import wandb
from Trifuse.utils import DEFAULT_CONFIG_FILE, RANK
from Trifuse.utils.misc import load_config
from Trifuse.utils.trainer import TriFuseTrainer


class TriFuse:
    def __init__(
        self,
        num_classes: int,
        head: str,
        variant: str = "tiny",
    ):
        self.num_classes = num_classes
        self.head = head
        self.variant = variant
        self.args = None
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
        self.args = load_config(config_path, overrides)
        self.args.num_classes = self.num_classes
        self.args.head = self.head
        self.args.variant = self.variant

        # Check dataset path
        assert (
            self.args.dataset_root is not None
        ), "Dataset path is not set. Please set it in the config file or pass it as an argument."

        # Initialize Wandb
        self.args.logger = None
        if self.args.enable_logger:
            print("[INFO] Initializing Wandb...")
            assert (
                os.environ.get("WANDB_API_KEY") is not None
            ), "Wandb API key is not set. Please set it in the environment variable WANDB_API_KEY."
            self._wandb_init()
        else:
            print("[INFO] Wandb logging is disabled.")

        # Initialize trainer
        print("[INFO] Initializing trainer...")
        self.trainer = TriFuseTrainer(vars(self.args))

        # Start training
        print("[INFO] Starting training...")
        self.trainer.train()

        # Done training
        print("[INFO] Training done.")

    def _wandb_init(self):
        if RANK in {-1, 0}:
            wandb.login(key=os.environ["WANDB_API_KEY"])
            self.args.logger = wandb.init(project="trifuse", config=vars(self.args))
            self.args.logger.define_metric("eval/precision", summary="max")
            self.args.logger.define_metric("eval/recall", summary="max")
            self.args.logger.define_metric("eval/mAP50", summary="max")
            self.args.logger.define_metric("eval/mAP5095", summary="max")

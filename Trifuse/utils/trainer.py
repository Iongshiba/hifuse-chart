import os
import subprocess
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from Trifuse.utils import RANK, USER_CONFIG_DIR
from Trifuse.utils.build import (build_criterion, build_dataloader,
                                 build_dataset, build_lr_scheduler,
                                 build_model)
from Trifuse.utils.dist import ddp_cleanup, generate_ddp_command
from Trifuse.utils.engine import evaluate, train_one_epoch


class TriFuseDetector(nn.Module):
    def __init__(self, num_classes: int, head_type: str, variant: str = "tiny"):
        super(TriFuseDetector, self).__init__()
        self.head_type = head_type
        self.backbone, self.head = build_model(num_classes, head_type, variant)

    def forward(self, images, targets=None):
        if self.head_type == "retina":
            return self.head(self.backbone(images), images, targets)
        return self.head(self.backbone(images))

    def compute_loss(self, outputs, targets, criterion):
        return self.head.compute_loss(outputs, targets, criterion)


class TriFuseTrainer:
    def __init__(self, args: dict):
        self.args = Namespace(**args)
        self.world_size = -1
        self.model = None
        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        self.device = self._device_init()
        if self.device.type == "cpu":
            self.args.num_workers = 0

    def train(self):
        if isinstance(self.args.device, str):
            self.world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (list, tuple)):
            self.world_size = len(self.args.device)
        else:
            self.world_size = 1 if torch.cuda.is_available() else 0

        if self.world_size > 1 and "LOCAL_RANK" not in os.environ:
            cmd, file = generate_ddp_command(self.world_size, self)
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            self._train()

    def _train(self):
        print(f"[TRAINER] Training TriFuse. I'm rank {RANK}.")

        if self.world_size > 1:
            self._ddp_init()
        self._train_init()

        # training loop
        for epoch in range(self.args.resume_epoch, self.args.epochs):
            train_loss = train_one_epoch(
                model=self.model,
                optimizer=self.optimizer,
                dataloader=self.train_loader,
                criterion=self.criterion,
                device=self.device,
                epoch=epoch,
                lr_scheduler=self.scheduler,
                global_rank=RANK,
                logger=self.args.logger,
                scaler=self.scaler,
                amp=self.args.amp,
                max_norm=self.args.max_norm,
            )

            if RANK in {-1, 0}:
                stats = evaluate(
                    model=self.model,
                    dataloader=self.val_loader,
                    device=self.device,
                    epoch=epoch,
                )

                if self.args.enable_logger:
                    self.args.logger.log(stats)

    def _ddp_init(self):
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            rank=RANK,
            world_size=self.world_size,
        )

    def _train_init(self):
        # Model
        print("[TRAINER] Building model...")
        self.model = TriFuseDetector(
            self.args.num_classes, self.args.head, self.args.variant
        ).to(self.device)

        # Freeze layers     TODO: What is this doing?
        if self.args.freeze_layers:
            for name, para in self.model.named_parameters():
                # All weights except head are frozen
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

        # Check AMP
        self.scaler = torch.amp.GradScaler(enabled=self.args.amp)

        # Dataloaders
        print("[TRAINER] Building datasets...")
        self.train_dataset, self.val_dataset = build_dataset(
            data_type=self.args.data_type,
            image_size=self.args.image_size,
            root_path=self.args.dataset_root,
            disable_bbox_transform=self.args.disable_bbox_transform,
        )

        print("[TRAINER] Building dataloaders...")
        self.train_loader = build_dataloader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            rank=RANK,
            num_workers=self.args.num_workers,
            seed=self.args.seed,
            pin_memory=self.args.pin_memory,
        )

        self.val_loader = build_dataloader(
            dataset=self.val_dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            rank=RANK,
            num_workers=self.args.num_workers,
            seed=self.args.seed,
            pin_memory=self.args.pin_memory,
        )

        # Optimizer
        if self.args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.args.optimizer} not implemented."
            )

        # Criterion
        self.criterion = build_criterion(
            num_classes=self.args.num_classes, head=self.args.head
        )

        # Scheduler
        self.scheduler = build_lr_scheduler(
            optimizer=self.optimizer,
            num_step=len(self.train_loader),
            epochs=self.args.epochs,
            warmup=self.args.warmup,
            warmup_epochs=self.args.warmup_epochs,
        )

        # Resume training
        if self.args.resume:
            self._load_checkpoint()

        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[RANK], find_unused_parameters=True
            )

    def _save_checkpoint(self, epoch):
        """
        Save model, optimizer, scheduler state as
        {root or USER_CONFIG_DIR}/checkpoint/checkpoint_epoch_{epoch}.pth
        """
        checkpoint = {
            "net": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "lr_scheduler": self.scheduler.state_dict(),
        }

        checkpoint_dir = (
            Path(self.args.root) if self.args.root else USER_CONFIG_DIR
        ) / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")
        print(f"[TriFuse] Saved checkpoint at epoch {epoch} to {checkpoint_dir}")

    def _find_checkpoint(self) -> Path:
        """
        Locate which .pth to load:
          - If resume_epoch > 0, pick that checkpoint.
          - Else pick the checkpoint with the highest epoch number.
        Raises if none found.
        """
        checkpoint_path = None

        checkpoint_dir = (
            Path(self.args.root) if self.args.root else USER_CONFIG_DIR
        ) / "checkpoint"
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"No checkpoint dir: {checkpoint_dir}")

        # User specified resume epoch
        if self.args.resume_epoch:
            checkpoint_path = (
                checkpoint_dir / f"checkpoint_epoch_{self.args.resume_epoch}.pth"
            )
        else:
            checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))

            def epoch_num(p: Path):
                return int(p.stem.rsplit("_", 1)[1])

            checkpoint_path = (
                checkpoint_dir
                / f"checkpoint_epoch_{max(checkpoints, key=epoch_num)}.pth"
            )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")

        return checkpoint_path

    def _load_checkpoint(self) -> int:
        """
        Load from the checkpoint returned by _find_checkpoint().
        Returns the next epoch to begin (loaded_epoch + 1).
        """
        ckpt_path = self._find_checkpoint()
        print(f"[TRAINER] Resuming from checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

        loaded_epoch = ckpt.get("epoch", 0)
        return loaded_epoch + 1

    def _save_weight(self, epoch: int) -> None:
        """
        Save only the best model weights as best.pt.
        """
        base_dir = Path(self.args.root) if self.args.root else USER_CONFIG_DIR
        weight_dir = base_dir / "weight"
        weight_dir.mkdir(parents=True, exist_ok=True)

        path = weight_dir / "best.pt"
        torch.save(self.model.state_dict(), path)
        print(f"[TriFuse] Saved best‐model at epoch {epoch}: {path}")

    def _load_weight(self) -> None:
        """
        Load the best.pt weights into the model.
        """
        base_dir = Path(self.args.root) if self.args.root else USER_CONFIG_DIR
        weight_path = base_dir / "weight" / "best.pt"

        if not weight_path.is_file():
            raise FileNotFoundError(f"No best.pt found at {weight_path}")

        print(f"[TRAINER] Loading best weights from: {weight_path}")
        state = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state)

    def _device_init(self) -> torch.device:
        """
        Initialize the device based on the config.
        """
        device_arg = self.args.device
        if device_arg == "cpu":
            return torch.device("cpu")
        elif device_arg == "cuda":
            return torch.device("cuda")
        elif device_arg is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            raise ValueError(f"Invalid device: {self.args.device}")

from argparse import Namespace
import subprocess
import os
import torch
import torch.distributed as dist
from Trifuse.utils import (
    RANK,
    LOCAL_RANK,
)
from Trifuse.utils.dist import (
    generate_ddp_command,
    ddp_cleanup,
)


class TriFuseTrainer:
    def __init__(self, args: dict):
        self.epochs = None
        self.batch_size = None
        self.save = None
        self.save_every = None
        self.device = None
        self.workers = None
        self.root = None
        self.optimizer = None
        self.enable_logger = None
        self.seed = None
        self.resume = None
        self.amp = None
        self.freeze = None
        self.dropout = None
        self.world_size = -1

        for k, v in args.items():
            setattr(self, k, v)

    def train(self):
        if isinstance(self.device, str):
            self.world_size = len(self.device.split(","))
        elif isinstance(self.device, (list, tuple)):
            self.world_size = len(self.device)
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
        if self.world_size > 1:
            self._ddp_init()
        self._train_init()

    def _ddp_init(self):
        torch.cuda.set_device(LOCAL_RANK)
        self.device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            rank=LOCAL_RANK,
            world_size=self.world_size,
        )

    def _train_init(self):
        # Model
        # Freeze layers
        # Check AMP
        # Dataloaders
        # Optimizer
        # Scheduler
        pass

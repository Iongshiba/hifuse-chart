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
    def __init__(self, args: Namespace):
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

        for k, v in vars(args).items():
            setattr(self, k, v)

    def train(self):
        if isinstance(self.device, str):
            world_size = len(self.device.split(","))
        elif isinstance(self.device, (list, tuple)):
            world_size = len(self.device)
        else:
            world_size = 1 if torch.cuda.is_available() else 0

        if world_size > 1:
            cmd, file = generate_ddp_command(world_size, self)
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            self._train()

    def _train(self):
        pass

    def _ddp_init(self, world_size: int):
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            rank=RANK,
            world_size=world_size,
        )

    def _train_init(self):
        pass

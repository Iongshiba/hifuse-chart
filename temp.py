import os
import sys
import math
import random
import yaml
import wandb
import numpy as np

from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from functools import partial
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from torch.nn.parallel import DistributedDataParallel as DDP

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class YoloFormatDataset(Dataset):
    def __init__(self, img_dir, transforms=None):
        self.transforms = transforms

        img_dir = Path(img_dir)
        if not img_dir.exists():
            raise ValueError(f"Path not found: {img_dir}")

        # If they gave us the train/ or valid/ folder, dive into its images/ subfolder
        if img_dir.is_dir() and (img_dir / "images").is_dir():
            images_path = img_dir / "images"
        else:
            images_path = img_dir

        # Collect all images with common extensions
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif")
        self.images = []
        for ext in exts:
            self.images += list(images_path.glob(ext))
            self.images += list(images_path.glob(ext.upper()))

        if len(self.images) == 0:
            raise ValueError(f"No images found in {images_path!r}")

        # Labels live alongside train/valid, in labels/ subfolder
        self.labels_dir = images_path.parent / "labels"
        if not self.labels_dir.is_dir():
            raise ValueError(f"Labels folder not found: {self.labels_dir!r}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        stem = img_path.stem
        label_path = self.labels_dir / f"{stem}.txt"

        # load image
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # parse YOLO-format bboxes
        boxes, labels = [], []
        if label_path.exists():
            for line in open(label_path):
                try:
                    c, xc, yc, w, h = tuple(map(float, line.strip().split()[:5]))
                except:
                    print(line)
                    raise ValueError()
                x1 = (xc - w / 2) * W
                y1 = (yc - h / 2) * H
                x2 = (xc + w / 2) * W
                y2 = (yc + h / 2) * H
                boxes.append([x1, y1, x2, y2])
                labels.append(int(c))

        if self.transforms:
            img = self.transforms(img)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


# Hyperparameters and paths
data_yaml = "/kaggle/input/bok-choy-disease-detection-yolo-format/data.yaml"
project_dir = Path("/kaggle/working/retinanet_bokchoy/")
train_dir = "/kaggle/input/bok-choy-disease-detection-yolo-format/train"
val_dir = "/kaggle/input/bok-choy-disease-detection-yolo-format/valid"
epochs = 100
batch_size = 16
img_size = 640
device_ids = [0, 1]
lr0 = 0.01
momentum = 0.937
weight_decay = 0.0005
dropout_p = 0.1
num_classes = 2

# Transform
transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
)

# Seed
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Output dirs
(project_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
(project_dir / "logs").mkdir(parents=True, exist_ok=True)

# Load data.yaml
with open(data_yaml) as f:
    data = yaml.safe_load(f)
class_names = data["names"]


# Custom Retina cls head with dropout
class RetinaClassificationHeadDropout(nn.Module):
    def __init__(
        self,
        in_channels,
        num_anchors,
        num_classes,
        num_convs=4,
        prior_prob=0.01,
        dropout=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
        self.cls_subnet = nn.Sequential(*layers)

        self.cls_score = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )

        for module in list(self.cls_subnet) + [
            self.cls_score
        ]:  # TODO: might switch to kaiming init? later
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                nn.init.constant_(module.bias, 0)

        # Set bias for focal loss as in the original paper
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        logits = []
        for x in features:
            t = self.cls_subnet(x)
            logits.append(self.cls_score(t))
        return logits


def main_worker(rank, world_size):
    # DDP init
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Seed set
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    # Dataset, DistributedSampler, DataLoader
    train_ds = YoloFormatDataset(train_dir, transforms=transform)
    val_ds = YoloFormatDataset(val_dir, transforms=transform)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Model wrap
    model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
    num_anchors = model.head.classification_head.num_anchors
    # replace head
    model.head.classification_head = RetinaClassificationHeadDropout(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        dropout=dropout_p,
    )
    model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Optimizer, LR scheduler
    optimizer = optim.SGD(
        model.parameters(), lr=lr0, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # WandB
    if rank == 0:
        from kaggle_secrets import UserSecretsClient
        import wandb

        wandb_key = UserSecretsClient().get_secret("WANDB_API_KEY")
        wandb.login(key=wandb_key)
        logger = wandb.init(
            project="mp242",
            config={"lr0": lr0, "epochs": epochs, "batch_size": batch_size},
        )
        logger.define_metric("eval/precision", summary="max")
        logger.define_metric("eval/recall", summary="max")
        logger.define_metric("eval/mAP50", summary="max")
        logger.define_metric("eval/mAP5095", summary="max")

    # Training loop
    best_map = -1.0

    for epoch in range(1, epochs + 1):
        # 1) epoch-level sampler shuffle
        train_sampler.set_epoch(epoch)

        # 2) Training
        model.train()
        torch.cuda.empty_cache()
        accu_loss = 0.0
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        else:
            pbar = train_loader

        for images, targets in pbar:
            images = [img.to(rank) for img in images]
            targets = [{k: v.to(rank) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accu_loss += loss.item()
            if rank == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}", lr=optimizer.param_groups[0]["lr"]
                )

                logger.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                )

        scheduler.step()

        # 3) Validation + COCOeval (only rank 0 needs to do it)
        if rank == 0:
            model.eval()
            all_dets, all_tgts = [], []

            with torch.no_grad():
                for images, targets in val_loader:
                    imgs_cuda = [img.to(rank) for img in images]
                    tgts_cuda = [{k: v.to(rank) for k, v in t.items()} for t in targets]

                    outputs = model(imgs_cuda)
                    # collect predictions
                    for out in outputs:
                        all_dets.append(
                            {
                                "boxes": out["boxes"].cpu(),
                                "scores": out["scores"].cpu(),
                                "labels": out["labels"].cpu(),
                            }
                        )
                    # collect GT
                    for t in targets:
                        all_tgts.append(
                            {"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()}
                        )

            # build coco_gt / coco_dt in memory (same as your snippet)
            coco_gt = {
                "images": [],
                "annotations": [],
                "categories": [{"id": i, "name": n} for i, n in enumerate(class_names)],
            }
            ann_id = 0
            for img_id, (tgt, img_path) in enumerate(zip(all_tgts, val_ds.images)):
                w, h = Image.open(img_path).size
                coco_gt["images"].append(
                    {"id": img_id, "width": w, "height": h, "file_name": img_path.name}
                )
                for box, lbl in zip(tgt["boxes"], tgt["labels"]):
                    x1, y1, x2, y2 = box.tolist()
                    ww, hh = x2 - x1, y2 - y1
                    coco_gt["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": int(lbl.item()),
                            "bbox": [x1, y1, ww, hh],
                            "area": ww * hh,
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

            cocoGt = COCO()
            cocoGt.dataset = coco_gt
            cocoGt.createIndex()

            coco_dt = []
            for img_id, det in enumerate(all_dets):
                for box, score, lbl in zip(det["boxes"], det["scores"], det["labels"]):
                    x1, y1, x2, y2 = box.tolist()
                    ww, hh = x2 - x1, y2 - y1
                    coco_dt.append(
                        {
                            "image_id": img_id,
                            "category_id": int(lbl.item()),
                            "bbox": [x1, y1, ww, hh],
                            "score": float(score.item()),
                        }
                    )

            cocoDt = cocoGt.loadRes(coco_dt)
            cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            stats = cocoEval.stats
            mAP5095 = float(stats[0])
            mAP50 = float(stats[1])
            recall_mat = cocoEval.eval["recall"]
            recall50 = recall_mat[0, :, :, 0]
            avg_recall = float(np.nanmean(recall50))
            prec_mat = cocoEval.eval["precision"]
            precision50 = prec_mat[0, :, :, 0, 2]
            avg_prec = float(np.nanmean(precision50))

            # log & checkpoint
            logger.log(
                {
                    "eval/mAP5095": mAP5095,
                    "eval/mAP50": mAP50,
                    "eval/recall": avg_recall,
                    "eval/precision": avg_prec,
                }
            )

            ckpt = {
                "epoch": epoch,
                "best_map": best_map,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }

            ckpt_path = project_dir / "checkpoints" / f"retina_ep{epoch:03d}.pth"
            torch.save(ckpt, ckpt_path)
            if mAP5095 > best_map:
                best_map = mAP5095
                torch.save(
                    model.module.state_dict(), project_dir / "checkpoints" / "best.pth"
                )

    if rank == 0:
        logger.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    # ChatGPT solution: "Force “fork” start method so spawn() can see main". Might dive deeper into this later
    mp.set_start_method("fork", force=True)
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

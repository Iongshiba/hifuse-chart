import sys

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from utils.misc import box_cxcywh_to_xywh, plot_bboxes_batch


def train_one_epoch(
    model,
    optimizer,
    dataloader,
    criterion,
    device,
    epoch,
    lr_scheduler,
    global_rank,
    logger,
    scaler,
    amp,
    max_norm,
):
    torch.cuda.empty_cache()
    model.train()
    if criterion is not None:
        criterion.train()
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    bar = tqdm(dataloader, file=sys.stdout, disable=global_rank not in {-1, 0})
    for step, data in enumerate(bar):
        images, targets, _ = data
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images, targets)
            loss_dict = model.compute_loss(logits, targets, criterion)
            loss = loss_dict["tloss"]

        scaler.scale(loss).backward()

        accu_loss += loss
        bar.desc = f"[Train Epoch {epoch}] Loss: {loss.item():.3f}\tLR: {optimizer.param_groups[0]['lr']:.6f}"
        if logger is not None and global_rank in {-1, 0}:
            logger.log(
                {"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"]}
            )

        scaler.step(optimizer)

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, dataloader, device, epoch):
    model.eval()

    images_coco = []
    coco_predictions = []

    global_idx = 0
    bar = tqdm(dataloader, file=sys.stdout, desc=f"Eval Epoch {epoch}")
    for images, anns, items in bar:
        batch_size = len(images)

        preds = model(images.to(device))

        for b in range(batch_size):
            info = items[b]
            img_id = info["id"]
            images_coco.append(info)

            pred = preds[b]
            boxes_xyxy = pred["boxes"]  # [N,4] (x1,y1,x2,y2)
            scores = pred["scores"]  # [N]
            labels = pred["labels"]  # [N]

            if boxes_xyxy.numel() == 0:
                continue

            # Convert to COCO xywh:
            # x = x1, y = y1, w = x2-x1, h = y2-y1
            x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
            widths = (x2 - x1).clamp(min=0)
            heights = (y2 - y1).clamp(min=0)
            boxes_xywh = torch.stack([x1, y1, widths, heights], dim=1)

            for box, score, label in zip(boxes_xywh, scores, labels):
                coco_predictions.append(
                    {
                        "id": global_idx,
                        "image_id": img_id,
                        "category_id": int(label.item()),
                        "bbox": box.cpu().tolist(),
                        "score": float(score.item()),
                    }
                )
                global_idx += 1

    # if no predictions, return early
    if len(coco_predictions) == 0:
        print("No predictions found.")
        return {
            "eval/precision": 0.0,
            "eval/recall": 0.0,
            "eval/mAP50": 0.0,
            "eval/mAP5095": 0.0,
        }

    # Load ground truth and detections
    gt_coco = COCO(dataloader.dataset.label_path)
    dt_coco = gt_coco.loadRes(coco_predictions)
    coco_eval = COCOeval(gt_coco, dt_coco, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Compute the aggregated metrics
    stats = coco_eval.stats
    mAP5095 = float(stats[0])  # AP@[.5:.95]
    mAP50 = float(stats[1])  # AP@.5
    recall = coco_eval.eval["recall"]
    recall50 = recall[0, :, :, 0]  # [R, K]
    avg_recall = float(np.nanmean(recall50))

    precision = coco_eval.eval["precision"]
    precision50 = precision[0, :, :, 0, 2]
    avg_precision = float(np.nanmean(precision50))

    return {
        "eval/precision": avg_precision,
        "eval/recall": avg_recall,
        "eval/mAP50": mAP50,
        "eval/mAP5095": mAP5095,
    }


@torch.no_grad()
def plot_img(model, dataset, device, args):
    model.eval()
    images = []
    pd_boxes = []
    gt_boxes = []

    for i in range(args.num_plot):
        image, ann, item = dataset[i]
        image = image.to(device)
        images.append(Image.open(item["file_path"]))
        img_w, img_h = item["width"], item["height"]

        pred = model(image.unsqueeze(0))

        gt_box = box_cxcywh_to_xywh(ann["boxes"])
        gt_box[:, [0, 1, 2, 3]] *= torch.tensor([img_w, img_h, img_w, img_h])
        gt_boxes.append(gt_box)

        pd_box = box_cxcywh_to_xywh(pred["pred_boxes"].squeeze().detach().cpu())
        pd_box[:, [0, 1, 2, 3]] *= torch.tensor([img_w, img_h, img_w, img_h])
        pd_boxes.append(pd_box)

    stats = {
        "image_with_bboxes": plot_bboxes_batch(
            images,
            pd_boxes,
            gt_boxes,
            args.num_plot,
        ),
    }

    return stats


def train_one_epoch_retina(
    model,
    optimizer,
    dataloader,
    criterion,
    device,
    epoch,
    lr_scheduler,
    global_rank,
    logger,
    scaler,
    amp,
    max_norm,
):

    torch.cuda.empty_cache()
    model.train()

    # criterion.train()

    accu_loss = torch.zeros(1).to(device)

    optimizer.zero_grad(set_to_none=True)

    bar = tqdm(dataloader, file=sys.stdout, disable=global_rank != 0)

    for step, data in enumerate(bar):
        images, anns, _ = data
        images = images.to(device)
        anns = [{k: v.to(device) for k, v in t.items()} for t in anns]

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss_dict = model(images, anns)
            # loss = loss_dict["loss"]
            cls_loss = loss_dict["classification"]
            bbox_loss = loss_dict["bbox_regression"]
            loss = cls_loss + bbox_loss

        scaler.scale(loss).backward()

        accu_loss += loss

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.update()
        optimizer.step()
        lr_scheduler.step()

        bar.desc = f"[Train Epoch {epoch}] Loss: {loss.item():.3f}\tLR: {optimizer.param_groups[0]['lr']:.6f}"

        if logger is not None and global_rank == 0:
            logger.log(
                {"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"]}
            )

    return accu_loss.item() / (step + 1)

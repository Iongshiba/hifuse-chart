import os
import sys
import torch

import numpy as np

from PIL import Image
from tqdm import tqdm
from utils.misc import box_cxcywh_to_xywh, plot_bboxes_batch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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
    criterion.train()
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    bar = tqdm(dataloader, file=sys.stdout, disable=global_rank != 0)
    for step, data in enumerate(bar):
        images, anns, _ = data
        images = images.to(device)
        anns = [{k: v.to(device) for k, v in t.items()} for t in anns]

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            preds = model(images)
            losses = criterion(preds, anns)
            loss = sum(losses.values())

        scaler.scale(loss).backward()

        accu_loss += loss
        bar.desc = f"[Train Epoch {epoch}] Loss: {loss.item():.3f}\tLR: {optimizer.param_groups[0]['lr']:.6f}"
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
    accu_loss = torch.zeros(1).to(device)
    out_logits = []
    out_bboxes = []
    images_coco = []

    bar = tqdm(dataloader, file=sys.stdout)
    for step, data in enumerate(bar):
        images, anns, item = data
        images = images.to(device)
        anns = [{k: v.to(device) for k, v in t.items()} for t in anns]

        preds = model(images)
        out_logits.append(preds["pred_logits"])
        out_bboxes.append(preds["pred_boxes"])
        images_coco += item

        # gt_labels.append([t["labels"] for t in anns])
        # gt_bboxes.append([t["boxes"] for t in anns])

    ### VALIDATE ###

    out_logits = torch.cat(out_logits, dim=0)
    out_bboxes = torch.cat(out_bboxes, dim=0)

    prob = out_logits.softmax(-1)
    # change the index if non_object_index changes
    # prob[..., 0] *= 0
    scores, labels = prob.max(-1)

    boxes = box_cxcywh_to_xywh(out_bboxes)
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = images_coco[0]["height"], images_coco[0]["width"]
    scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=out_bboxes.device)
    boxes = boxes * scale_fct[None, None, :]

    ### EVALUATION WITH PYCOCOTOOLS ###

    predictions = [
        {
            "id": i,
            "image_id": images_coco[i]["id"],
            "category_id": l.tolist(),
            "bbox": b.tolist(),
            "score": s.tolist(),
        }
        for i in range(len(images_coco))
        for l, b, s in zip(labels[i], boxes[i], scores[i])
    ]

    gt_coco = COCO(dataloader.dataset.label_path)
    pd_coco = gt_coco.loadRes(predictions)
    eval_coco = COCOeval(gt_coco, pd_coco, "bbox")

    eval_coco.evaluate()
    eval_coco.accumulate()

    precision = eval_coco.eval["precision"]
    recall = eval_coco.eval["recall"]
    precision_50_all = precision[0, :, :, 0, 2]  # IoU=0.5, all categories, all areas
    avg_precision = np.mean(precision_50_all[precision_50_all > 0])  # Skip NaN values
    # Get max recall at IoU=0.5 for all categories and areas
    recall_50_all = recall[0, :, 0, 2]  # IoU=0.5, all categories, all areas
    avg_recall = np.mean(recall_50_all[recall_50_all > 0])  # Skip NaN values

    eval_coco.summarize()

    stats = {
        "eval/precision": avg_precision.item(),
        "eval/recall": avg_recall.item(),
        "eval/mAP50": eval_coco.stats[1].item(),
        "eval/mAP5095": eval_coco.stats[0].item(),
    }

    print(
        f"[Val Epoch {epoch}]\tPrecision: {avg_precision.item()}\tRecall: {avg_recall.item()}\tmAP@.5: {eval_coco.stats[1].item()}\tmAP@[.5:.95]: {eval_coco.stats[0].item()}"
    )

    return stats


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


@torch.no_grad()
def evaluate(model, dataloader, device, logger):
    model.eval()
    torch.cuda.empty_cache()

    from ultralytics.utils.metrics import DetMetrics

    metrics = DetMetrics()

    bar = tqdm(dataloader, file=sys.stdout)

    for step, data in enumerate(bar):
        images, anns, _ = data
        images = images.to(device, non_blocking=True)
        anns = [{k: v.to(device) for k, v in t.items()} for t in anns]

        preds = model(images)

        for i, pred in enumerate(preds):
            if i >= len(anns):
                continue  # Skip if no target for this prediction

            target = anns[i]

            boxes = pred["boxes"]
            scores = pred["scores"]
            labels = pred["labels"]

            # Skip if no detections
            if len(boxes) == 0:
                continue

            # Convert boxes from [x1, y1, x2, y2] to [x1, y1, w, h]
            # No need for conversion if the model already outputs in [x1, y1, x2, y2] format

            # Process target format - assuming it's in YOLO format
            target_boxes = None
            target_cls = None

            # Determine target format and extract necessary data
            if isinstance(target, dict):
                # Handle dict format targets
                if "boxes" in target and "labels" in target:
                    target_boxes = target["boxes"].to(device)
                    target_cls = target["labels"].to(device)
            elif isinstance(target, torch.Tensor):
                # Handle YOLO format: [class_id, x_center, y_center, width, height] or another format
                if target.size(-1) >= 5:
                    target_cls = target[:, 0].to(device)
                    target_boxes = target[:, 1:5].to(device)

            if target_boxes is None or target_cls is None:
                continue  # Skip if target format can't be determined

            # Convert predictions to format needed by Ultralytics' process method
            # Calculate IoU between predictions and targets
            ious = box_iou(boxes, target_boxes)
            correct = torch.zeros(len(boxes), device=device)  # init tp tensor

            if len(target_boxes) > 0:
                # For each prediction, identify if it matches a ground truth
                for j, pred_box in enumerate(boxes):
                    pred_label = labels[j]

                    # Find potential matches (same class)
                    same_class = target_cls == pred_label

                    if not same_class.any():
                        continue  # No matching class in ground truth

                    # Get IoUs for this prediction with ground truths of same class
                    valid_ious = ious[j][same_class]

                    if len(valid_ious) > 0:
                        best_iou, best_idx = valid_ious.max(0)

                        # If IoU exceeds threshold, this is a true positive
                        if best_iou >= iou_threshold:
                            correct[j] = 1

            # Now convert tensors to numpy arrays for DetMetrics.process()
            tp = correct.cpu().numpy()
            conf = scores.cpu().numpy()
            pred_cls = labels.cpu().numpy()
            target_cls = target_cls.cpu().numpy()

            # Update metrics using Ultralytics DetMetrics
            metrics.process(tp, conf, pred_cls, target_cls)

            # Update progress bar description periodically
            if i % 10 == 0:
                current_map = metrics.results_dict.get("map50-95", 0)
                bar.desc = f"[Validate] mAP: {current_map:.4f}"

    # Get final metrics
    results = metrics.results_dict

    # Log metrics in the same format as train_one_epoch
    if logger is not None:
        for k, v in results.items():
            logger.log({f"val/{k}": v})

    return results


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
            loss = model(images, anns)["loss"]

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

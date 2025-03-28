import os
import sys
import torch

import numpy as np

from PIL import Image
from tqdm import tqdm
from utils.misc import box_cxcywh_to_xywh
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def train_one_epoch(
    model, optimizer, dataloader, criterion, device, epoch, lr_scheduler
):
    model.train()
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    bar = tqdm(dataloader, file=sys.stdout)
    for step, data in enumerate(bar):
        images, anns, _ = data
        images = images.to(device)
        anns = [{k: v.to(device) for k, v in t.items()} for t in anns]
        sample_num += images.shape[0]

        preds = model(images)
        losses = criterion(preds, anns)
        loss = sum(losses.values())

        loss.backward()
        accu_loss += loss

        bar.desc = f"[Train Epoch {epoch}] Loss: {loss.item():.3f}\tLR: {optimizer.param_groups[0]['lr']:.6f}"
        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training ", loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, epoch):
    model.eval()
    accu_loss = torch.zeros(1).to(device)
    out_logits = []
    out_bboxes = []
    images_coco = []

    sample_num = 0
    bar = tqdm(dataloader, file=sys.stdout)
    for step, data in enumerate(bar):
        images, anns, item = data
        images = images.to(device)
        anns = [{k: v.to(device) for k, v in t.items()} for t in anns]
        sample_num += images.shape[0]

        preds = model(images.to(device))
        out_logits.append(preds["pred_logits"])
        out_bboxes.append(preds["pred_boxes"])
        images_coco += item
        # gt_labels.append([t["labels"] for t in anns])
        # gt_bboxes.append([t["boxes"] for t in anns])

    out_logits = torch.cat(out_logits, dim=0)
    out_bboxes = torch.cat(out_bboxes, dim=0)

    prob = out_logits.softmax(-1)
    # change the index if non_object_index changes
    prob[..., 0] *= 0
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
        "precision": avg_precision.item(),
        "recall": avg_recall.item(),
        "mAP50": eval_coco.stats[1].item(),
        "mAP5095": eval_coco.stats[0].item(),
    }

    print(
        f"[Val Epoch {epoch}]\tPrecision: {avg_precision.item()}\tRecall: {avg_recall.item()}\tmAP@.5: {eval_coco.stats[1].item()}\tmAP@[.5:.95]: {eval_coco.stats[0].item()}"
    )

    return stats

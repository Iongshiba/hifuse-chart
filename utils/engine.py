import os
import sys
import torch
from utils.misc import evaluate_yolo_with_coco
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


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

        dataloader.desc = f"[Train Epoch {epoch}]\tLoss: {loss.item():.3f}\tLR: {optimizer.param_groups[0]['lr']:.6f}"
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
    pd_labels = []
    pd_bboxes = []
    gt_labels = []
    gt_bboxes = []

    sample_num = 0
    bar = tqdm(dataloader, file=sys.stdout)
    for step, data in enumerate(bar):
        images, anns, images_id = data
        images = images.to(device)
        anns = [{k: v.to(device) for k, v in t.items()} for t in anns]
        sample_num += images.shape[0]

        print(images_id)
        preds = model(images.to(device))
        pd_labels += preds["pred_logits"]
        pd_bboxes += preds["pred_boxes"]
        gt_labels.append([t["labels"] for t in anns])
        gt_bboxes.append([t["boxes"] for t in anns])

        if step == 3:
            break

        # dataloader.desc = f"[Validate Epoch {epoch}]\tLoss: {loss.item():.3f}"

    pd_labels = torch.stack(pd_labels, dim=0)
    pd_bboxes = torch.stack(pd_bboxes, dim=0)

    result = evaluate_yolo_with_coco(
        dataloader.dataset.base_dir,
        dataloader.dataset.class_file,
        (pd_bboxes, pd_labels),
        images_id,
    )

    print(result)

    return accu_loss.item() / (step + 1)


def retina_train_one_epoch(
    model, dataloader, optimizer, scheduler, device, epoch, print_freq=50
):
    model.train()
    total_cls_loss = 0.0
    total_reg_loss = 0.0

    progress_bar = tqdm(dataloader, file=sys.stdout, desc=f"Epoch {epoch}")

    for batch_idx, data in enumerate(progress_bar):
        images, targets = data
        images = torch.stack(images).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        cls_loss, reg_loss = loss_dict["cls_loss"], loss_dict["reg_loss"]

        loss = cls_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()

        if batch_idx % print_freq == 0:
            progress_bar.set_description(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                f"Cls Loss: {cls_loss.item():.4f} | Reg Loss: {reg_loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_reg_loss = total_reg_loss / len(dataloader)
    print(
        f"Epoch [{epoch}] Completed - Avg Cls Loss: {avg_cls_loss:.4f} | Avg Reg Loss: {avg_reg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
    )
    return avg_cls_loss, avg_reg_loss


@torch.no_grad()
def retina_evaluate(model, dataloader, coco_gt, device):
    model.eval()
    results = []

    for images, targets in dataloader:
        images = torch.stack(images).to(device)
        predictions = model(images)

        for img_id, pred in zip(
            [t["image_id"].cpu().numpy().tolist() for t in targets], predictions
        ):
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                x_min, y_min, x_max, y_max = box
                results.append(
                    {
                        "image_id": int(img_id),
                        "category_id": int(label),
                        "bbox": [
                            round(float(x_min), 2),
                            round(float(y_min), 2),
                            round(float(x_max - x_min), 2),
                            round(float(y_max - y_min), 2),
                        ],
                        "score": round(float(score), 3),
                    }
                )

    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]  # mAP

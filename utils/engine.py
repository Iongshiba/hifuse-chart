import os
import sys
import torch
from utils.misc import evaluate_yolo_with_coco
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

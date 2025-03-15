import sys
import torch

from tqdm import tqdm


def train_one_epoch(
    model, optimizer, dataloader, criterion, device, epoch, lr_scheduler
):
    model.train()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    dataloader = tqdm(dataloader, file=sys.stdout)
    for step, data in enumerate(dataloader):
        images, labels = data
        images = images.to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
        sample_num += images.shape[0]

        preds = model(images)
        losses = criterion(preds, labels)
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
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    dataloader = tqdm(dataloader, file=sys.stdout)
    for step, data in enumerate(dataloader):
        images, labels = data
        images = images.to(device)
        labels = [{k: v.to(device) for k, v in t} for t in labels]
        sample_num += images.shape[0]

        preds = model(images.to(device))
        losses = criterion(preds, labels)
        loss = sum(losses.values())

        accu_loss += loss

        dataloader.desc = f"[Validate Epoch {epoch}]\tLoss: {loss.item():.3f}"

    return accu_loss.item() / (step + 1)

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import torch

import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from utils.build import (
    TriFuse_Tiny,
    get_params_groups,
    create_lr_scheduler,
    create_criterion,
    create_dataset,
)
from utils.engine import train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    print(args)
    print(
        'Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/'
    )

    tb_writer = SummaryWriter()

    batch_size = args.batch_size
    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, 8]
    )  # number of workers
    print("Using {} dataloader workers every process".format(nw))

    train_dataset, val_dataset = create_dataset(args)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )

    model = TriFuse_Tiny(num_classes=args.num_classes).to(device)
    criterion = create_criterion(num_classees=args.num_classes, head=args.head).to(
        device
    )

    if args.RESUME == False:
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(
                args.weights
            )
            weights_dict = torch.load(args.weights, map_location=device)["state_dict"]

            # Delete the weight of the relevant category
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            model.load_state_dict(weights_dict, strict=False)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # All weights except head are frozen
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(
        optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=1
    )

    best_map = -1.0
    start_epoch = 0

    if args.RESUME:
        path_checkpoint = "./model_weight/checkpoint/ckpt_best_100.pth"
        print("model continue train")
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        lr_scheduler.load_state_dict(checkpoint["lr_schedule"])

    for epoch in range(start_epoch + 1, args.epochs + 1):

        # train
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
        )

        # validate
        stats = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        tags = [
            "train_loss",
            "precision",
            "recall",
            "mAP50",
            "mAP5095",
            "learning_rate",
        ]

        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], stats["precision"], epoch)
        tb_writer.add_scalar(tags[2], stats["recall"], epoch)
        tb_writer.add_scalar(tags[3], stats["mAP50"], epoch)
        tb_writer.add_scalar(tags[4], stats["mAP5095"], epoch)
        tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)

        if best_map < stats["mAP5095"]:
            if not os.path.isdir("./model_weight"):
                os.mkdir("./model_weight")
            torch.save(model.state_dict(), "./model_weight/best_model.pth")
            print("Saved epoch{} as new best model".format(epoch))
            best_map = stats["mAP5095"]

        if epoch % 10 == 0:
            print("epoch:", epoch)
            print("learning rate:", optimizer.state_dict()["param_groups"][0]["lr"])
            checkpoint = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "lr_schedule": lr_scheduler.state_dict(),
            }
            if not os.path.isdir("./model_weight/checkpoint"):
                os.mkdir("./model_weight/checkpoint")
            torch.save(
                checkpoint, "./model_weight/checkpoint/ckpt_best_%s.pth" % (str(epoch))
            )

        # add loss, acc and lr into tensorboard
        print(
            f"[epoch {epoch}] precision: {stats['precision']:.2f} recall: {stats['recall']:.2f} mAP@.5: {stats['mAP50']:.2f} mAP@[.5:.95]: {stats['mAP5095']:.2f}"
        )

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--head", type=str, default="detr")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--RESUME", type=bool, default=False)

    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--root-data-path", type=str, default="")
    parser.add_argument("--train-data-path", type=str, default="")
    parser.add_argument("--val-data-path", type=str, default="")

    parser.add_argument("--weights", type=str, default="", help="initial weights path")

    parser.add_argument("--freeze-layers", type=bool, default=False)
    parser.add_argument(
        "--device", default="cuda:0", help="device id (i.e. 0 or 0,1 or cpu)"
    )

    opt = parser.parse_args()

    main(opt)

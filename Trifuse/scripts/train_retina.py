import argparse
import os

import torch
import torch.optim as optim
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.models.detection.retinanet import AnchorGenerator, RetinaNet

import wandb
# from torch.utils.tensorboard import SummaryWriter
# from models.retina import RetinaNet
from models.trifuse import TriFuse
from utils.build import (TriFuse_Tiny, create_dataset, create_lr_scheduler,
                         get_params_groups)
from utils.engine import evaluate_retina, plot_img, train_one_epoch_retina
from utils.misc import check_model_memory


@record
def main(args):

    ###########################
    ##                       ##
    ##    Data Paralleism    ##
    ##                       ##
    ###########################

    assert torch.cuda.is_available(), "Training on CPU is not supported"

    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])

        init_process_group(backend="nccl")
    else:
        local_rank = 0
        global_rank = 0

    assert local_rank != -1, "LOCAL_RANK environment variable not set"
    assert global_rank != -1, "RANK environment variable not set"

    # torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")

    # Set random seed for reproducibility
    # seed = 42 + global_rank
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    print(f"GPU {local_rank} - Using device: {device}")

    ##############################
    ##                          ##
    ##   Wandb Initialization   ##
    ##                          ##
    ##############################

    logger = None
    if global_rank == 0 or not args.distributed and args.enable_logger:
        print("Wandb logger is enabled. Beginning wandb initialization...")
        wandb.login(key=os.environ["WANDB_API_KEY"])
        logger = wandb.init(project="trifuse", config=args)
        logger.define_metric("eval/precision", summary="max")
        logger.define_metric("eval/recall", summary="max")
        logger.define_metric("eval/mAP50", summary="max")
        logger.define_metric("eval/mAP5095", summary="max")

    ###################
    ##               ##
    ##    Dataset    ##
    ##               ##
    ###################

    batch_size = args.batch_size
    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, 8]
    )  # number of workers
    print("Using {} dataloader workers every process".format(nw))

    train_dataset, val_dataset = create_dataset(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(not args.distributed),
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,
        sampler=(
            DistributedSampler(train_dataset, shuffle=True)
            if args.distributed
            else None
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )

    #################
    ##             ##
    ##    Model    ##
    ##             ##
    #################

    # model = TriFuse_Tiny(num_classes=args.num_classes, head="retina").to(device)
    anchor_sizes = tuple(
        (s, int(s * 2 ** (1 / 3)), int(s * 2 ** (2 / 3)))
        for s in [32, 64, 128, 256]  # only 4 levels!
    )
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)  # length = 4

    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    backbone = TriFuse(
        depths=(2, 2, 6, 2),
        conv_depths=(2, 2, 6, 2),
        num_classes=args.num_classes,
        out_channels=256,
        head=args.head,
    )
    model = RetinaNet(
        backbone=backbone,
        num_classes=args.num_classes,
        min_size=224,
        max_size=224,
        anchor_generator=anchor_generator,
    ).to(device)
    model = (
        DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
        if args.distributed
        else model
    )
    criterion = None
    pg = get_params_groups(model, weight_decay=args.wd, learning_rate=args.lr)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(
        optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=1
    )
    # Mixed Precision
    scaler = torch.amp.GradScaler(enabled=args.amp)

    ####################
    ##                ##
    ##    Training    ##
    ##                ##
    ####################

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

    best_map = -1.0
    start_epoch = 0

    if args.RESUME:
        assert args.root_path != "", "checkpoint path is None when resume training"
        assert args.resume_epoch != -1, "resume epoch is None when resume training"
        path_checkpoint = (
            args.root_path
            + "/model_weight/checkpoint/ckpt_best_%s.pth" % (str(args.resume_epoch))
        )
        print("model continue train")
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        lr_scheduler.load_state_dict(checkpoint["lr_schedule"])

    for epoch in range(start_epoch + 1, args.epochs + 1):

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch_retina(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            lr_scheduler=lr_scheduler,
            global_rank=global_rank if args.distributed else 0,
            logger=logger,
            scaler=scaler,
            amp=args.amp,
            max_norm=args.max_norm,
        )

        if global_rank == 0 or not args.distributed:
            stats = evaluate_retina(
                model=model,
                dataloader=val_loader,
                device=device,
                epoch=epoch,
            )

            if logger is not None:
                logger.log(stats)

            if best_map < stats["eval/mAP5095"]:
                if not os.path.isdir(args.root_path + "/model_weight/"):
                    os.mkdir(args.root_path + "/model_weight/")
                torch.save(
                    model.state_dict(),
                    args.root_path + "/model_weight/best_model.pth",
                )
                print("Saved epoch{} as new best model".format(epoch))
                best_map = stats["eval/mAP5095"]

            if epoch % args.save_every == 0:
                checkpoint = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "lr_schedule": lr_scheduler.state_dict(),
                }
                if not os.path.isdir(args.root_path + "/model_weight/checkpoint"):
                    os.mkdir(args.root_path + "/model_weight/checkpoint")
                torch.save(
                    checkpoint,
                    args.root_path
                    + "/model_weight/checkpoint/ckpt_best_%s.pth" % (str(epoch)),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--enable-logger", action="store_false", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)

    args = parser.parse_args()

    print(args)

    main(args)

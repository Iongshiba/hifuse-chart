import argparse

from Trifuse.models.trifuse import TriFuse
from Trifuse.utils import DEFAULT_CONFIG_FILE


def main(args):
    model = TriFuse(
        head=args.head,
        num_classes=args.num_classes,
        variant=args.variant,
    )
    model.train(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=str)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--variant", type=str, default="tiny")

    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--dataset-root", type=str, default="")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data-type", type=str, default="coco")
    parser.add_argument("--disable-bbox-transform", action="store_true", default=False)
    parser.add_argument("--enable-logger", action="store_false", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--max-norm", type=float, default=0.1)
    # parser.add_argument("--pin-memory", action="store_true", default=False)

    args = parser.parse_args()

    print(args)

    main(args)

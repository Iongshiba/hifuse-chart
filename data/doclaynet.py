import os
import json
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class YOLODataset(Dataset):
    def __init__(
        self,
        images_path: list,
        labels_path: list,
        num_classes=1,
        transform=None,
    ):
        self.image_dir = os.path.dirname(images_path[0])
        self.label_dir = os.path.dirname(labels_path[0])
        self.base_dir = os.path.dirname(self.image_dir)
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        img_w, img_h = img.size
        if img.mode != "RGB":
            img = img.convert("RGB")

        anns = self._load_label(self.labels_path[item], img_w, img_h)
        if len(anns) == 0:
            label = {
                "labels": torch.as_tensor([self.num_classes], dtype=torch.int64),
                "boxes": torch.zeros(1, 4, dtype=torch.float32),
            }
        else:
            label = {
                "labels": torch.as_tensor([box[0] for box in anns], dtype=torch.int64),
                "boxes": torch.as_tensor(
                    [box[1:5] for box in anns], dtype=torch.float32
                ),
            }

        if self.transform is not None:
            img = self.transform(img)

        return img, label, self.images_path[item]

    def _load_label(self, label_path, img_w, img_h):
        with open(label_path, "r") as f:
            lines = f.readlines()

        boxes = []
        for line in lines:
            parts = line.strip().split()
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            width = float(parts[3]) * img_w
            height = float(parts[4]) * img_h
            class_id = int(parts[0])

            # Append the box as a tuple
            boxes.append([class_id, x_center, y_center, width, height])

        return boxes

    @staticmethod
    def collate_fn(batch):
        images, labels, paths = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = list(labels)
        paths = list(paths)
        return images, labels, paths


class COCODataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_path: str,
        num_classes=1,
        transform=None,
    ):
        self.image_dir = image_dir
        self.label_path = label_path
        self.annotations = self._load_annotations(label_path)
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.annotations["images"])

    def __getitem__(self, item):
        img_item = self.annotations["images"][item]
        img_item["file_path"] = os.path.join(self.image_dir, img_item["file_name"])
        img_path = os.path.join(self.image_dir, img_item["file_name"])
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        anns = self._get_label(img_item["id"])
        if len(anns) == 0:
            label = {
                "labels": torch.zeros((0,), dtype=torch.int64),
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
            }
        else:
            label = {
                "labels": torch.as_tensor([box[0] for box in anns], dtype=torch.int64),
                "boxes": torch.as_tensor(
                    [box[1:5] for box in anns], dtype=torch.float32
                ),
            }
        if self.transform is not None:
            img, label = self.transform(img, label)

        return img, label, img_item

    def _load_annotations(self, label_path):
        with open(label_path, "r") as f:
            data = json.load(f)

        return data

    def _get_label(self, img_id):
        labels = [
            label
            for label in self.annotations["annotations"]
            if label["image_id"] == img_id
        ]
        boxes = []
        for label in labels:
            box = label["bbox"]
            class_id = int(label["category_id"])
            x_min = float(box[0])
            y_min = float(box[1])
            width = float(box[2])
            height = float(box[3])

            x_center = x_min + width / 2
            y_center = y_min + height / 2

            x_max = x_min + width
            y_max = y_min + height
            if x_max > x_min and y_max > y_min:
                boxes.append([class_id, x_min, y_min, x_max, y_max])

        return boxes

    @staticmethod
    def collate_fn(batch):
        images, labels, items = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = list(labels)
        items = list(items)
        return images, labels, items


class YOLOCOCODataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform if transform is not None else self._transform()

        self.img_files = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        self.img_files.sort()

    def __len__(self):
        return len(self.img_files)

    def _transform(self):
        return Compose(
            [
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_label(self, label_path, img_width, img_height):
        boxes = []
        labels = []

        with open(label_path, "r") as f:
            for line in f.readlines():
                if line.strip():
                    data = line.strip().split()
                    class_id = int(data[0])
                    cx = float(data[1]) * img_width
                    cy = float(data[2]) * img_height
                    w = float(data[3]) * img_width
                    h = float(data[4]) * img_height

                    x1 = cx - (w / 2)
                    y1 = cy - (h / 2)
                    x2 = cx + (w / 2)
                    y2 = cy + (h / 2)

                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            # Return empty tensors if no boxes
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.as_tensor([-1], dtype=torch.int64)

        return boxes, labels

    def __getitem__(self, idx):
        """Get image and corresponding labels"""
        img_path = self.img_files[idx]

        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.label_dir, f"{base_name}.txt")

        img = Image.open(img_path).convert("RGB")

        width, height = img.size
        boxes, labels = self._load_label(label_path, width, height)

        img = self.transform(img)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        return img, target

    @staticmethod
    def collate_fn(batch):
        images = []
        targets = []

        for img, target in batch:
            images.append(img)
            targets.append(target)

        # Stack images
        images = torch.stack(images, dim=0)

        return images, targets

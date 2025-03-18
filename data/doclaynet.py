import os
import torch

from PIL import Image
from torch.utils.data import Dataset


class YOLODataset(Dataset):
    def __init__(
        self,
        images_path: list,
        labels_path: list,
        class_file: str,
        num_classes=1,
        transform=None,
    ):
        self.class_file = class_file
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
        if img.mode != "RGB":
            img = img.convert("RGB")

        anns = self._load_label(self.labels_path[item])
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

    def _load_label(self, label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        boxes = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
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

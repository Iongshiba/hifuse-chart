import torch

from PIL import Image
from torch.utils.data import Dataset


class YOLODataset(Dataset):
    def __init__(self, images_path: list, labels_path: list, transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != "RGB":
            img = img.convert("RGB")

        boxes = self._load_label(self.labels_path[item])
        label = {
            "labels": [box[0] for box in boxes],
            "boxes": [box[1:5] for box in boxes],
        }

        if self.transform is not None:
            img = self.transform(img)

        return img, label

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

        return torch.tensor(boxes)

import os
import sys
import copy
import json
import math
import torch
import pickle
import random

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from config import SUPPORTED

import torch.nn as nn
import matplotlib.pyplot as plt


def read_train_data_classification(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    category = [
        cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))
    ]
    category.sort()
    class_indices = dict((k, v) for v, k in enumerate(category))
    json_str = json.dumps(
        dict((val, key) for key, val in class_indices.items()), indent=4
    )

    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []

    for cls in category:
        cls_path = os.path.join(root, cls)
        images = [
            os.path.join(root, cls, i)
            for i in os.listdir(cls_path)
            if os.path.splitext(i)[-1] in SUPPORTED
        ]

        image_class = class_indices[cls]

        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)

    print("{} images for training.".format(len(train_images_path)))

    return train_images_path, train_images_label


def read_val_data_classification(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    category = [
        cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))
    ]
    category.sort()
    class_indices = dict((k, v) for v, k in enumerate(category))

    val_images_path = []
    val_images_label = []

    for cls in category:
        cls_path = os.path.join(root, cls)
        images = [
            os.path.join(root, cls, i)
            for i in os.listdir(cls_path)
            if os.path.splitext(i)[-1] in SUPPORTED
        ]
        image_class = class_indices[cls]

        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)

    print("{} images for validation.".format(len(val_images_path)))

    return val_images_path, val_images_label


def read_train_data_detection(root: str):
    random.seed(0)
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    images_dir = os.path.join(root, "images", "train")
    labels_dir = os.path.join(root, "labels", "train")

    assert os.path.exists(images_dir), f"images directory: {images_dir} does not exist."
    assert os.path.exists(labels_dir), f"labels directory: {labels_dir} does not exist."

    train_images_path = []
    train_labels_path = []

    image_files = [
        f for f in os.listdir(images_dir) if os.path.splitext(f)[-1] in SUPPORTED
    ]

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(label_path):
            img_path = os.path.join(images_dir, img_file)
            train_images_path.append(img_path)
            train_labels_path.append(label_path)

    print(f"{len(train_images_path)} images for training.")

    return train_images_path, train_labels_path


def read_val_data_detection(root: str):
    random.seed(0)
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    images_dir = os.path.join(root, "images", "val")
    labels_dir = os.path.join(root, "labels", "val")

    assert os.path.exists(images_dir), f"images directory: {images_dir} does not exist."
    assert os.path.exists(labels_dir), f"labels directory: {labels_dir} does not exist."

    val_images_path = []
    val_labels_path = []

    image_files = [
        f for f in os.listdir(images_dir) if os.path.splitext(f)[-1] in SUPPORTED
    ]

    for img_file in image_files:
        # Get corresponding label file (same name but .txt extension)
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        label_path = os.path.join(labels_dir, label_file)

        # Only include images that have corresponding label files
        if os.path.exists(label_path):
            img_path = os.path.join(images_dir, img_file)
            val_images_path.append(img_path)
            val_labels_path.append(label_path)

    print(f"{len(val_images_path)} images for validation.")

    return val_images_path, val_labels_path


class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

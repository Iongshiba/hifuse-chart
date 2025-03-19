import os
import json
import random


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

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cls in category:
        cls_path = os.path.join(root, cls)
        images = [
            os.path.join(root, cls, i)
            for i in os.listdir(cls_path)
            if os.path.splitext(i)[-1] in supported
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

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cls in category:
        cls_path = os.path.join(root, cls)
        images = [
            os.path.join(root, cls, i)
            for i in os.listdir(cls_path)
            if os.path.splitext(i)[-1] in supported
        ]
        image_class = class_indices[cls]

        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)

    print("{} images for validation.".format(len(val_images_path)))

    return val_images_path, val_images_label


def read_data_detection_yolo(root: str):
    random.seed(0)
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    train_images_dir = os.path.join(root, "images", "train")
    train_labels_dir = os.path.join(root, "labels", "train")
    val_images_dir = os.path.join(root, "images", "val")
    val_labels_dir = os.path.join(root, "labels", "val")

    assert os.path.exists(
        train_images_dir
    ), f"images directory: {train_images_dir} does not exist."
    assert os.path.exists(
        train_labels_dir
    ), f"labels directory: {train_labels_dir} does not exist."
    assert os.path.exists(
        val_images_dir
    ), f"images directory: {val_images_dir} does not exist."
    assert os.path.exists(
        val_labels_dir
    ), f"labels directory: {val_labels_dir} does not exist."

    train_images_path = []
    train_labels_path = []
    val_images_path = []
    val_labels_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    train_image_files = [
        f for f in os.listdir(train_images_dir) if os.path.splitext(f)[-1] in supported
    ]

    for img_file in train_image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        label_path = os.path.join(train_labels_dir, label_file)

        if os.path.exists(label_path):
            img_path = os.path.join(train_images_dir, img_file)
            train_images_path.append(img_path)
            train_labels_path.append(label_path)

    val_image_files = [
        f for f in os.listdir(val_images_dir) if os.path.splitext(f)[-1] in supported
    ]

    for img_file in val_image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        label_path = os.path.join(val_labels_dir, label_file)

        if os.path.exists(label_path):
            img_path = os.path.join(val_images_dir, img_file)
            val_images_path.append(img_path)
            val_labels_path.append(label_path)

    print(f"{len(train_images_path)} images for training.")
    print(f"{len(val_images_path)} images for validation.")

    return train_images_path, train_labels_path, val_images_path, val_labels_path


def read_data_detection_coco(root: str):
    random.seed(0)
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    train_images_dir = os.path.join(root, "train")
    val_images_dir = os.path.join(root, "val")
    train_label_path = os.path.join(root, "train.json")
    val_label_path = os.path.join(root, "val.json")

    assert os.path.exists(
        train_images_dir
    ), f"images directory: {train_images_dir} does not exist."
    assert os.path.exists(
        val_images_dir
    ), f"images directory: {val_images_dir} does not exist."
    assert os.path.exists(
        train_label_path
    ), f"images directory: {train_label_path} does not exist."
    assert os.path.exists(
        val_label_path
    ), f"images directory: {val_label_path} does not exist."

    return train_images_dir, train_label_path, val_images_dir, val_label_path

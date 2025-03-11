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


def read_train_data_detection(root: str):
    random.seed(0)
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    images_dir = os.path.join(root, "images", "train")
    labels_dir = os.path.join(root, "labels", "train")

    assert os.path.exists(images_dir), f"images directory: {images_dir} does not exist."
    assert os.path.exists(labels_dir), f"labels directory: {labels_dir} does not exist."

    train_images_path = []
    train_labels_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    image_files = [
        f for f in os.listdir(images_dir) if os.path.splitext(f)[-1] in supported
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

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    image_files = [
        f for f in os.listdir(images_dir) if os.path.splitext(f)[-1] in supported
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

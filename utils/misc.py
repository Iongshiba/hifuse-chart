import os
import copy
import json
import glob
import torch
import pickle
import numpy as np
import torch.nn as nn

from PIL import Image
from torchvision.ops.boxes import box_area
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch.nn as nn
import matplotlib.pyplot as plt


def write_pickle(list_info: list, file_name: str):
    with open(file_name, "wb") as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, "rb") as f:
        info_list = pickle.load(f)
        return info_list


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]

    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]

    return torch.stack(b, dim=-1)


def generalized_iou(boxes1, boxes2):
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = default_iou(boxes1, boxes2)

    tl = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    br = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (br - tl).clamp(min=0)

    area = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (area - union) / area

    return giou


def default_iou(boxes1, boxes2):
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # [num, 4] -> matrix of iou pair-wise between boxes
    tl = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    br = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # [num_box1, num_box2, 2]
    wh = (br - tl).clamp(min=0)

    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def get_clones(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


def confusion_matrix(x, y, num_classes):
    indices = num_classes * y + x
    matrix = torch.bincount(indices, minlength=num_classes**2).view(
        num_classes, num_classes
    )
    return matrix


def convert_yolo_to_coco(base_dir, yolo_class_file, output_json, split="val"):
    # Read class names
    with open(yolo_class_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    images_dir = os.path.join(base_dir, f"images/{split}")
    labels_dir = os.path.join(base_dir, f"labels/{split}")

    images = []
    annotations = []
    categories = []

    # Create categories
    for i, class_name in enumerate(class_names):
        categories.append({"id": i, "name": class_name, "supercategory": "none"})

    annotation_id = 1

    # Process each image
    for img_id, img_path in enumerate(glob.glob(os.path.join(images_dir, "*.*"))):
        if not (
            img_path.endswith(".jpg")
            or img_path.endswith(".png")
            or img_path.endswith(".jpeg")
        ):
            continue

        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        img_id = base_name

        # Get image dimensions
        img = Image.open(img_path)
        width, height = img.size

        # Add image info
        images.append(
            {"id": img_id, "file_name": img_name, "width": width, "height": height}
        )

        # Find corresponding annotation file
        ann_path = os.path.join(labels_dir, f"{base_name}.txt")

        if not os.path.exists(ann_path):
            continue

        # Process annotations
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])

                x_center, y_center = float(parts[1]), float(parts[2])
                bbox_width, bbox_height = float(parts[3]), float(parts[4])

                # Convert to COCO format
                x_min = (x_center - bbox_width / 2) * width
                y_min = (y_center - bbox_height / 2) * height
                bbox_width_abs = bbox_width * width
                bbox_height_abs = bbox_height * height

                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": class_id,
                        "bbox": [x_min, y_min, bbox_width_abs, bbox_height_abs],
                        "area": bbox_width_abs * bbox_height_abs,
                        "segmentation": [],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

    # Create COCO dataset
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    # Save to JSON file
    with open(output_json, "w") as f:
        json.dump(coco_format, f)

    return output_json


def evaluate_yolo_with_coco(
    base_dir, class_file, predictions, image_paths, obj_threshold=0.5, split="val"
):
    gt_coco_json = convert_yolo_to_coco(
        base_dir=base_dir,
        yolo_class_file=class_file,
        output_json=f"gt_{split}_coco.json",
        split=split,
    )

    coco_gt = COCO(gt_coco_json)

    coco_predictions = []
    prediction_id = 1
    pd_bboxes, pd_labels = predictions

    for i in range(len(pd_labels)):
        img_id = os.path.basename(image_paths[i])
        img_width, img_height = Image.open(image_paths[i]).size

        for pd_bbox, pd_label in zip(pd_bboxes[i], pd_labels[i]):
            print(pd_bboxes[i])
            print(pd_labels[i])
            x_center, y_center, width, height = pd_bbox
            score = pd_label.softmax(dim=-1)
            confidence = score
            class_id = 0 if score > obj_threshold else 1

            x_min = (x_center - width / 2) * img_width
            y_min = (y_center - height / 2) * img_height
            bbox_width = width * img_width
            bbox_height = height * img_height

            coco_predictions.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(class_id),
                    "bbox": [
                        float(x_min),
                        float(y_min),
                        float(bbox_width),
                        float(bbox_height),
                    ],
                    "score": float(confidence),
                    "id": prediction_id,
                }
            )
            prediction_id += 1

    # Create prediction object
    coco_dt = coco_gt.loadRes(coco_predictions)

    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = "./class_indices.json"
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, "r")
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype("uint8"))
        plt.show()


if __name__ == "__main__":
    yolo_label_D0r
    convert_yolo_to_coco()

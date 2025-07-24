import torch
from torchvision.models.detection import image_list
from Trifuse.models.retina import RetinaNet

debug_dict = torch.load("debug_head_outputs.pt")

head = RetinaNet(
    num_classes=2,
)

head_outputs = {
    "cls_logits": debug_dict["cls_logits"],
    "bbox_regression": debug_dict["bbox_regression"],
}
anchors = debug_dict["anchors"]
image_list = debug_dict["image_list"]

head.eval()
preds = head._inference(head_outputs, anchors, image_list.image_sizes)

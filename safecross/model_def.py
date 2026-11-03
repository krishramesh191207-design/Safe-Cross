"""Model definition shared by the export script.

We take torchvision's COCO-pretrained SSDLite320-MobileNetV3-Large and keep only
the tensor-in / tensor-out part (normalise -> backbone -> SSD head -> softmax).
Anchor decoding and NMS are done in NumPy at runtime (see postprocess.py),
because data-dependent ops like NMS do not lower cleanly to ExecuTorch/XNNPACK.
"""
import numpy as np
import torch
from torch import nn
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.image_list import ImageList

INPUT_SIZE = 320


class SSDLiteCore(nn.Module):
    """Input: float32 [1, 3, 320, 320] RGB in [0, 1].
    Output: (box_regression [1, A, 4], class_scores [1, A, 91]) with A = 3234 anchors."""

    def __init__(self, det_model):
        super().__init__()
        self.backbone = det_model.backbone
        self.head = det_model.head

    def forward(self, x):
        x = (x - 0.5) / 0.5  # same normalisation as torchvision's transform
        feats = list(self.backbone(x).values())
        out = self.head(feats)
        return out["bbox_regression"], torch.softmax(out["cls_logits"], dim=-1)


def load_pretrained():
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    det = ssdlite320_mobilenet_v3_large(weights=weights).eval()
    return det, weights.meta["categories"]


def compute_anchors(det_model) -> np.ndarray:
    """Default boxes (xyxy, in 320x320 pixel space) for a 320x320 input."""
    core = SSDLiteCore(det_model).eval()
    x = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        feats = list(core.backbone(x).values())
        anchors = det_model.anchor_generator(
            ImageList(x, [(INPUT_SIZE, INPUT_SIZE)]), feats
        )[0]
    return anchors.numpy().astype(np.float32)

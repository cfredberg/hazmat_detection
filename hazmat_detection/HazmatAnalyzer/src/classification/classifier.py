import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.config import (
    CLASSIFIER_WEIGHTS, CLF_IMG_SIZE, CLF_BACKBONE,
    HAZARD_CLASSES, COLOR_CLASSES, SYMBOL_CLASSES,
)
from src.classification.model import build_model

_TRANSFORM = transforms.Compose([
    transforms.Resize((CLF_IMG_SIZE, CLF_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class HazmatClassifier:
    def __init__(self, weights=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = build_model(
            n_classes=len(HAZARD_CLASSES),
            n_colors=len(COLOR_CLASSES),
            n_symbols=len(SYMBOL_CLASSES),
            backbone=CLF_BACKBONE,
        ).to(self.device)

        ckpt = torch.load(weights or CLASSIFIER_WEIGHTS, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def classify(self, roi):
        if isinstance(roi, np.ndarray):
            roi = Image.fromarray(roi[..., ::-1])
        tensor = _TRANSFORM(roi).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(tensor)
        elapsed = time.perf_counter() - t0

        def _decode(logits, labels):
            probs = torch.softmax(logits, dim=1)[0]
            idx   = probs.argmax().item()
            return labels[idx], float(probs[idx])

        cls, cls_conf = _decode(outputs["class"],  HAZARD_CLASSES)
        col, col_conf = _decode(outputs["color"],  COLOR_CLASSES)
        sym, sym_conf = _decode(outputs["symbol"], SYMBOL_CLASSES)

        return {
            "hazard_class": cls,
            "color": col,
            "symbol": sym,
            "confidence": round((cls_conf + col_conf + sym_conf) / 3, 4),
            "conf_class": round(cls_conf, 4),
            "conf_color": round(col_conf, 4),
            "conf_symbol": round(sym_conf, 4),
            "inference_time_ms": round(elapsed * 1000, 2),
        }

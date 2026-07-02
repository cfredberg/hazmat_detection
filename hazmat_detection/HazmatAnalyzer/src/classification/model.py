import torch
import torch.nn as nn
import timm


class MultiHeadHazmatClassifier(nn.Module):
    def __init__(self, n_classes, n_colors, n_symbols,
                 backbone="efficientnet_b0", dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        feat_dim      = self.backbone.num_features

        def _head(n_out):
            return nn.Sequential(
                nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(256, n_out),
            )

        self.class_head  = _head(n_classes)
        self.color_head  = _head(n_colors)
        self.symbol_head = _head(n_symbols)

    def forward(self, x):
        feats = self.backbone(x)
        return {
            "class":  self.class_head(feats),
            "color":  self.color_head(feats),
            "symbol": self.symbol_head(feats),
        }


def build_model(n_classes, n_colors, n_symbols, backbone="efficientnet_b0"):
    return MultiHeadHazmatClassifier(n_classes, n_colors, n_symbols, backbone)

"""End-to-end pipeline: YOLO detection → ROI crop → EfficientNet classification."""

import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import HAZARD_CLASSES
from src.detection.detector       import HazmatDetector
from src.classification.classifier import HazmatClassifier


class HazmatPipeline:
    def __init__(self, detector_weights=None, classifier_weights=None, det_conf=0.25):
        self.detector   = HazmatDetector(weights=detector_weights, conf=det_conf)
        self.classifier = HazmatClassifier(weights=classifier_weights)

    def run(self, image_path):
        """
        Run full pipeline on a single image.
        Returns structured dict.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        t_start = time.perf_counter()
        detections, det_time = self.detector.detect(image_path)

        results        = []
        clf_time_total = 0.0
        for roi_bgr, det in self.detector.crop_rois(img_bgr, detections):
            clf_out = self.classifier.classify(roi_bgr)
            clf_time_total += clf_out["inference_time_ms"]
            results.append({
                "bounding_box":   det["box"],
                "det_confidence": round(det["conf"], 4),
                "det_class_id":   det["class_id"],
                "det_class_name": HAZARD_CLASSES[det["class_id"]],
                "hazard_class":   clf_out["hazard_class"],
                "color":          clf_out["color"],
                "symbol":         clf_out["symbol"],
                "confidence":     clf_out["confidence"],
                "conf_class":     clf_out["conf_class"],
                "conf_color":     clf_out["conf_color"],
                "conf_symbol":    clf_out["conf_symbol"],
            })

        total_time = (time.perf_counter() - t_start) * 1000
        return {
            "image":          os.path.basename(image_path),
            "num_detections": len(results),
            "detections":     results,
            "timing_ms": {
                "detection":      round(det_time * 1000, 2),
                "classification": round(clf_time_total, 2),
                "total":          round(total_time, 2),
            },
        }

"""YOLO-based hazmat placard detector."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from configs.config import DETECTOR_WEIGHTS, YOLO_IMGSZ


class HazmatDetector:
    def __init__(self, weights=None, conf=0.25, iou=0.45):
        from ultralytics import YOLO
        self.model = YOLO(weights or DETECTOR_WEIGHTS)
        self.conf  = conf
        self.iou   = iou

    def detect(self, image):
        """
        Args:
            image: file path (str) or numpy HWC BGR array
        Returns:
            detections: list of {box, conf, class_id}
            elapsed:    seconds
        """
        t0 = time.perf_counter()
        results = self.model.predict(
            image, conf=self.conf, iou=self.iou,
            imgsz=YOLO_IMGSZ, verbose=False,
        )
        elapsed = time.perf_counter() - t0

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                detections.append({
                    "box":      [int(x1), int(y1), int(x2), int(y2)],
                    "conf":     float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                })
        return detections, elapsed

    def crop_rois(self, image_np, detections):
        """Return list of (roi_array, detection_dict) pairs."""
        crops = []
        h, w  = image_np.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if x2 > x1 and y2 > y1:
                crops.append((image_np[y1:y2, x1:x2], det))
        return crops

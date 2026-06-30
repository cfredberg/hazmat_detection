import cv2
from ultralytics import YOLO

from .HazmatAnalyzer.src.detection.detector import HazmatDetector
from .HazmatAnalyzer.src.classification.classifier import HazmatClassifier

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, CompressedImage

from rclpy.qos import QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge

import easyocr

from rapidfuzz import fuzz


# model = YOLO("classifier_best.pt")

# import torch
# import requests
# from io import BytesIO

# HF_URL = "https://huggingface.co/thalostech2025/thalos-hazmat-safety-v1/resolve/main/hazmat_safety_weights.pt"

# response = requests.get(HF_URL)
# model = torch.load(BytesIO(response.content), map_location="cpu")

HAZARD_CLASSES = [
    "poison", "oxygen", "flammable", "flammable-solid", "corrosive",
    "dangerous", "non-flammable-gas", "organic-peroxide", "explosive",
    "radioactive", "inhalation-hazard", "spontaneously-combustible",
    "infectious-substance",
]

class HazmatDetect(Node):
    def __init__(self):
        super().__init__("hazmat_detection")
        self.declare_parameter('use_webcam', False)
        self.declare_parameter('camera_id', 0)
        
        use_webcam = self.get_parameter('use_webcam').get_parameter_value().bool_value
        if not use_webcam:
            self.camera_id = self.get_parameter('camera_id').get_parameter_value().integer_value
        else:
            self.camera_id = 0

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.camera_subscription = self.create_subscription(
            CompressedImage,
            f'/cameras/raw/camera_{self.camera_id}',
            self.listener_callback,
            qos)

        self.threshold_sub = self.create_subscription(
            Float32,
            f'/hazmat/threshold',
            self.get_threshold,
            1)

        self.threshold = 0.95

        self.bridge = CvBridge()

        self.hazmat_frame_publisher = self.create_publisher(CompressedImage, f'/cameras/hazmat/camera_{self.camera_id}', qos)
        # self.hazmat_string_publisher = self.create_publisher(String, f'/hazmat/string/camera_{self.camera_id}', 1)

        self.detector = HazmatDetector()
        self.classifier = HazmatClassifier()

        self.reader = easyocr.Reader(['en'])

    def detect(self, frame):
        detections, elapsed = self.detector.detect(frame)

        # print(detections)

        for roi_bgr, det in self.detector.crop_rois(frame, detections):
            clf_out = self.classifier.classify(roi_bgr)
            results = {
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
            }
            detected_text = None
            if results["confidence"] > .92:
                if results["hazard_class"] == "dangerous":
                    detected_text = "dangerous_when_wet"
                elif results["hazard_class"] == "explosive":
                    text = self.best_ocr(roi_bgr).lower()
                    print(text)
                    scores = {
                        "explosives": fuzz.partial_ratio(text, "explosives"),
                        "blasting_agent": fuzz.partial_ratio(text, "blasting agent")
                    }

                    final_class = max(scores, key=scores.get)
                    if scores[final_class] > 0.97:
                        print(f"final: {final_class} conf: {scores[final_class]}")
                        detected_text = final_class
                elif results["hazard_class"] == "oxygen":
                    text = self.best_ocr(roi_bgr).lower()
                    print(text)
                    scores = {
                        "oxygen": fuzz.partial_ratio(text, "oxygen"),
                        "oxidizer": fuzz.partial_ratio(text, "oxidizer")
                    }

                    final_class = max(scores, key=scores.get)
                    if scores[final_class] > 0.97:
                        print(f"final: {final_class} conf: {scores[final_class]}")
                        detected_text = final_class
                elif results["hazard_class"] == "flammable":
                    text = self.best_ocr(roi_bgr).lower()
                    print(text)
                    scores = {
                        "flammable-gas": fuzz.partial_ratio(text, "flammable-gas"),
                        "fuel-oil": fuzz.partial_ratio(text, "fuel-oil")
                    }

                    final_class = max(scores, key=scores.get)
                    if scores[final_class] > 0.97:
                        print(f"final: {final_class} conf: {scores[final_class]}")
                        detected_text = final_class
                else:
                    detected_text = results["hazard_class"]
                
            if detected_text != None:
                results["hazard_class"] = detected_text
                cv2.rectangle(frame, (det["box"][0], det["box"][1]), (det["box"][2], det["box"][3]), (255, 0, 0), 3)
                cv2.putText(frame, f'{results["hazard_class"]}, {results["confidence"]}', (det["box"][0] + 2, det["box"][1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            # print(results)
        return frame

    def listener_callback(self, frame_msg):
        frame = self.bridge.compressed_imgmsg_to_cv2(frame_msg, desired_encoding='bgr8')

        # results = self.model(source=frame, conf=self.threshold)
        annotated_frame = self.detect(frame)

        new_frame_msg = self.bridge.cv2_to_compressed_imgmsg(annotated_frame, dst_format='jpg')
        new_frame_msg.header.stamp = self.get_clock().now().to_msg()
        self.hazmat_frame_publisher.publish(new_frame_msg)

        # detected_classes = list(set([results[0].names[int(cls)] for cls in results[0].boxes.cls]))
        # str_msg = String()
        # str_msg.data = detected_classes.__str__()
        # self.hazmat_string_publisher.publish(str_msg)

    def get_threshold(self, msg):
        self.threshold = msg.data

    def best_ocr(self, crop):
        angles = [0, 90, 180, 270]
        best_text = ""
        best_conf = 0.0

        for angle in angles:
            # Rotate properly
            if angle == 0:
                rotated = crop.copy()
            elif angle == 90:
                rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated = cv2.rotate(crop, cv2.ROTATE_180)
            elif angle == 270:
                rotated = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

            results = self.reader.readtext(rotated, detail=1)

            if not results:
                continue

            # Sort by vertical position and merge
            if isinstance(results[0], (list, tuple)):
                results = sorted(results, key=lambda r: r[0][0][1])
                merged_text = " ".join([r[1] for r in results])
                avg_conf = sum([r[2] for r in results]) / len(results)
            else:
                merged_text = " ".join(results) if isinstance(results, list) else str(results)
                avg_conf = 0.0

            # Keep the best result across rotations
            if avg_conf > best_conf:
                best_conf = avg_conf
                best_text = merged_text

        return best_text



def main(args=None):
    rclpy.init(args=args)

    hazmat_detect = HazmatDetect()

    rclpy.spin(hazmat_detect)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    hazmat_detect.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
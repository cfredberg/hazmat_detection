import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
from message_filters import Subscriber, TimeSynchronizer

import numpy as np

from pynput import keyboard

import cv2
import easyocr

import json

from cv_bridge import CvBridge

from Levenshtein import distance

class HazmatDetectNode(Node):

    def __init__(self):
        super().__init__('hazmat_detect')
        self.declare_parameter('use_webcam', False)
        self.declare_parameter('camera_name', "")
        
        use_webcam = self.get_parameter('use_webcam').get_parameter_value().bool_value
        if not use_webcam:
            self.camera_name = self.get_parameter('camera_name').get_parameter_value().string_value
        else:
            self.camera_name = 0

        self.camera_subscription = self.create_subscription(
            Image,
            f'/cameras/raw/camera_{self.camera_name}',
            self.listener_callback,
            1)

        self.bridge = CvBridge()

        self.reader = easyocr.Reader(["en"])

        self.hazmat_publisher = self.create_publisher(Image, f'/cameras/hazmat/camera_{self.camera_name}', 1)
        self.hazmat_string_publisher = self.create_publisher(String, f'/hazmat/string/camera_{self.camera_name}', 1)

        self.available_warnings = [
            "fuel oil",
            "explosives",
            "poison",
            "oxidizer",
            "corrosive",
            "combustible",
            "oxygen",
            "radioactive",
            "flammable",
            "blasting agent",
            "non-flammable gas",
            "inhalation hazard",
            "infectious substance",
            "flammable liquid",
            "flammable solid",
            "spontaneously combustible",
            "dangerous",
            "organic peroxide",
            "flammable gas",
        ]

        self.levenshtein_max_dist = 2
        self.levenshtein_min_confidence = 0.65
        self.sub_word_levenshtein_min_confidence = 0.5

        self.detection_proximity_margin = 200

    def listener_callback(self, frame_msg):
        # Get frames and display them
        frame = self.bridge.imgmsg_to_cv2(frame_msg, "bgr8")

        text = self.reader.readtext(frame)
        text_copy = text
        text = []

        detected_text = []

        for detect in text_copy:
            try:
                float(detect[1])
            except ValueError as e:
                text.append(detect)

        text.sort(key = lambda x: x[0][0][1])

        for detect in text:
            print(detect[1] + " " + str(detect[2] * 100 // 1) + "%")
            if detect[2] > self.levenshtein_min_confidence:
                detect = self.find_surrounding_labels(detect, text)
                dist = 100
                real_text = ""
                for warning in self.available_warnings:
                    new_dist = distance(detect[1].lower(), warning)
                    print(f"distance from {detect[1].lower()} to {warning} is {new_dist}")
                    if new_dist < dist:
                        dist = new_dist
                        real_text = warning
                if real_text != "" and dist <= self.levenshtein_max_dist:
                    cv2.rectangle(frame, (int(detect[0][0][0]), int(detect[0][0][1])), (int(detect[0][2][0]), int(detect[0][2][1])), (255, 0, 0), 2)
                    print(f"Found: {real_text}")
                    detected_text.append(real_text)

            # cv2.rectangle(frame, (int(detect[0][0][0]), int(detect[0][0][1])), (int(detect[0][2][0]), int(detect[0][2][1])), (255, 0, 0), 2)
            # print(f"Found: {detect[1]} with {int(detect[2]*100)}% confidence")

        print("-"*10)
        new_frame_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.hazmat_publisher.publish(new_frame_msg)

        str_msg = String()
        str_msg.data = str(detected_text)
        self.hazmat_string_publisher.publish(str_msg)
    
    def find_surrounding_labels(self, found_detect, all_text):
        found_detect = list(found_detect)
        found_text = found_detect[1]
        for detect in all_text:
            if detect[2] > self.sub_word_levenshtein_min_confidence:
                if detect[1] != found_text:
                    found_x = found_detect[0][0][0]
                    found_y = found_detect[0][0][1]
                    found_w = found_detect[0][2][0] - found_x
                    found_h = found_detect[0][2][1] - found_y

                    detect_x = detect[0][0][0]
                    detect_y = detect[0][0][1]
                    detect_w = detect[0][2][0] - detect_x
                    detect_h = detect[0][2][1] - detect_y

                    if detect_x > found_x - self.detection_proximity_margin and detect_x < found_x + found_w + self.detection_proximity_margin:
                        if detect_y > found_y - self.detection_proximity_margin/2 and detect_y < found_y + found_h + self.detection_proximity_margin:
                            found_detect[1] += f" {detect[1]}"
                            new_x1 = found_x
                            new_y1 = found_y
                            new_x2 = found_w + found_x
                            new_y2 = found_h + found_y

                            if detect_x < found_x:
                                new_x1 = detect_x
                            if detect_y < found_y:
                                new_y1 = detect_y
                            if detect_x+detect_w > found_x+found_w:
                                new_x2 = detect_x+detect_w
                            if detect_y+detect_h > found_y+found_h:
                                new_y2 = detect_y+detect_w

                            found_detect[0][0][0] = new_x1
                            found_detect[0][0][1] = new_y1
                            found_detect[0][2][0] = new_x2
                            found_detect[0][2][1] = new_y2
        return found_detect

def main(args=None):
    rclpy.init(args=args)

    hazmat_detect_node = HazmatDetectNode()

    rclpy.spin(hazmat_detect_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    hazmat_detect_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
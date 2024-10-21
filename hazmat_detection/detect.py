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
            10)

        self.bridge = CvBridge()

        self.reader = easyocr.Reader(["en"])

        self.hazmat_publisher = self.create_publisher(Image, f'/cameras/hazmat/camera_{self.camera_name}', 1)


    def listener_callback(self, frame_msg):
        # Get frames and display them
        frame = self.bridge.imgmsg_to_cv2(frame_msg, "bgr8")
        
        text = self.reader.readtext(frame)
        print(text)

        new_frame_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.hazmat_publisher.publish(new_frame_msg)

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
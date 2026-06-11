import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import Image

from ultralytics import YOLO


from cv_bridge import CvBridge


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

        self.camera_subscription = self.create_subscription(
            Image,
            f'/cameras/raw/camera_{self.camera_id}',
            self.listener_callback,
            1)

        self.threshold_sub = self.create_subscription(
            Float32,
            f'/hazmat/threshold',
            self.get_threshold,
            1)

        self.threshold = 0.95

        self.bridge = CvBridge()

        self.model = YOLO("/home/ros/ros2_ws/src/hazmat_detection/hazmat_model.pt")

        self.hazmat_frame_publisher = self.create_publisher(Image, f'/cameras/hazmat/camera_{self.camera_id}', 1)
        self.hazmat_string_publisher = self.create_publisher(String, f'/hazmat/string/camera_{self.camera_id}', 1)

    def listener_callback(self, frame_msg):
        frame = self.bridge.imgmsg_to_cv2(frame_msg, "bgr8")

        results = self.model(source=frame, conf=self.threshold)

        annotated_frame = results[0].plot()

        new_frame_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
        self.hazmat_frame_publisher.publish(new_frame_msg)

        detected_classes = list(set([results[0].names[int(cls)] for cls in results[0].boxes.cls]))
        str_msg = String()
        str_msg.data = detected_classes.__str__()
        self.hazmat_string_publisher.publish(str_msg)

    def get_threshold(self, msg):
        self.threshold = msg.data


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
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np


class ParkingLotDetectorNode(Node):
    def __init__(self):
        super().__init__('parking_lot_detector')
        self.subscription = self.create_subscription(
            Image,
            'image',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(PointStamped, 'parking_point', 10)
        self.bridge = CvBridge()
        self.get_logger().info('Parking lot detector node has been started.')

    def image_callback(self, msg):
        try:
            # Convert ROS image message to CV2 image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            # Perform the detection
            center_x, center_y = self.detect_parking_lot(cv_image)
            # Publish the center point as 3D point
            self.publish_center_point(center_x, center_y)
        except Exception as e:
            self.get_logger().error('Failed to process image: %r' % (e,))

    def detect_parking_lot(self, image):
        gray = image
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_x, center_y = 0, 0
        max_area = 0
        # Use the maximal contour to calculate the x and y
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                # Calculate the moment 
                M = cv2.moments(contour)
                # Use Moment to calculate the x and y
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])

        # Return the calculated center_x and center_y
        return center_x, center_y

    def publish_center_point(self, center_x, center_y):
        # Assume a fixed Z coordinate, for example 0
        point = PointStamped()
        point.header.stamp = self.get_clock().now().to_msg()
        point.header.frame_id = 'world'
        point.point.x = center_x
        point.point.y = center_y
        point.point.z = 100  # Fixed Z-coordinate
        self.publisher.publish(point)


def main(args=None):
    rclpy.init(args=args)
    parking_lot_detector_node = ParkingLotDetectorNode()
    rclpy.spin(parking_lot_detector_node)
    parking_lot_detector_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

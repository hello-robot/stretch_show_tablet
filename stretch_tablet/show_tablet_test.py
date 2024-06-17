import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import sophuspy as sp
import numpy as np
from scipy.spatial.transform import Rotation as R

from stretch_tablet.src.kinematics import Human, TabletPlanner, load_bad_json_data

class ShowTabletNode(Node):
    def __init__(self):
        super().__init__("show_tablet_node")

        # sub
        self.sub_face_landmarks = self.node.create_subscription(
            String,
            "/faces/landmarks_3d",
            callback=self.callback_face_landmarks,
            qos_profile=1
        )

        self.sub_body_landmarks = self.node.create_subscription(
            String,
            "/body_landmarks/landmarks_3d",
            callback=self.callback_body_landmarks,
            qos_profile=1
        )

        # tf
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(buffer=self.tf_buffer, node=self.node)

        # state
        self.face_landmarks = None
        self.body_landmarks = None
        self.latest_camera_pose = None
        self.planner = TabletPlanner()
        self.human = Human()

    # callbacks
    def callback_face_landmarks(self, msg: String):
        data = load_bad_json_data(msg.data)
        self.human.pose_estimate.set_face_estimate(data)

    def callback_body_landmarks(self, msg: String):
        data = load_bad_json_data(msg.data)
        self.human.pose_estimate.set_body_estimate(data)
    
    def clear_landmarks(self):
        self.human.pose_estimate.clear_estimates()

    # main
    def main(self):

        while rclpy.ok():
            if self.human.pose_estimate.is_populated():
                print('hi')

                try:
                    t = self.tf_buffer.lookup_transform(
                            "camera_color_optical_frame",
                            "odom",
                            rclpy.time.Time())
                    
                    camera_pos = t.transform.translation
                    camera_ori = t.transform.rotation
                    self.latest_camera_pose = [[camera_pos.x, camera_pos.y, camera_pos.z],
                                               [camera_ori.x, camera_ori.y, camera_ori.z, camera_ori.w]]
                except Exception as ex:
                    print(ex)                
            
            rclpy.spin_once(self.node, timeout_sec=0.1)

def main():
    ShowTabletNode().main()

if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import sophuspy as sp
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

# from stretch_tablet.src.kinematics import Human, TabletPlanner, load_bad_json_data
from .kinematics import Human, TabletPlanner, load_bad_json_data

class ShowTabletNode(Node):
    def __init__(self):
        super().__init__("show_tablet_node")

        # sub
        self.sub_face_landmarks = self.create_subscription(
            String,
            "/faces/landmarks_3d",
            callback=self.callback_face_landmarks,
            qos_profile=1
        )

        self.sub_body_landmarks = self.create_subscription(
            String,
            "/body_landmarks/landmarks_3d",
            callback=self.callback_body_landmarks,
            qos_profile=1
        )

        # tf
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(buffer=self.tf_buffer, node=self)
        # time.sleep(1.)

        # state
        self.face_landmarks = None
        self.body_landmarks = None
        self.latest_camera_pose = None
        self.planner = TabletPlanner()
        self.human = Human()

    # callbacks
    def callback_face_landmarks(self, msg: String):
        msg_data = msg.data.replace("\"", "")
        if msg_data == "None" or msg_data is None:
            return

        data = load_bad_json_data(msg_data)

        if data is None:
            return
        self.human.pose_estimate.set_face_estimate(data)

    def callback_body_landmarks(self, msg: String):
        msg_data = msg.data.replace("\"", "")
        if msg_data == "None" or msg_data is None:
            return

        data = load_bad_json_data(msg_data)

        if data is None or data == "{}":
            return
        self.human.pose_estimate.set_body_estimate(data)
    
    def clear_landmarks(self):
        self.human.pose_estimate.clear_estimates()

    def go_to_pose(self, pose):
        pass

    # main
    def main(self):
        while rclpy.ok():
            try:
                t = self.tf_buffer.lookup_transform(
                        "camera_color_optical_frame",
                        "odom",
                        rclpy.time.Time())
                
                camera_pos = t.transform.translation
                camera_ori = t.transform.rotation
                # self.latest_camera_pose = [[camera_pos.x, camera_pos.y, camera_pos.z],
                #                             [camera_ori.x, camera_ori.y, camera_ori.z, camera_ori.w]]
                camera_position = np.array([camera_pos.x, camera_pos.y, camera_pos.z])
                camera_orientation = R.from_quat([camera_ori.x, camera_ori.y, camera_ori.z, camera_ori.w]).as_matrix()
                world2camera_pose = sp.SE3(camera_orientation, camera_position)
                self.human.pose_estimate.set_camera_pose(world2camera_pose)

                if self.human.pose_estimate.is_populated():
                    # update ik
                    tablet_pose = self.planner.in_front_of_eyes(human=self.human)
                    ik_soln = self.planner.ik(tablet_pose)
                    print(ik_soln)

                    # clear buffer
                    self.human.pose_estimate.clear_estimates()

            except Exception as ex:
                print(ex)
            
            rclpy.spin_once(self, timeout_sec=0.1)

def main():
    rclpy.init()
    ShowTabletNode().main()

if __name__ == '__main__':
    main()
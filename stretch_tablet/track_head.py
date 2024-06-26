import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from .kinematics import load_bad_json_data, TabletController, Human
import json

def in_range(value, range):
    return True if value >= range[0] and value <= range[1] else False

class HeadTracker(Node):
    def __init__(self):
        super().__init__('head_tracker_node')

        # pub
        self.pub_tablet_move_by = self.create_publisher(
            String,
            "/stretch_tablet/move_by",
            qos_profile=1
        )

        # sub
        self.sub_face_landmarks = self.create_subscription(
            String,
            "/faces_gripper/landmarks_3d",
            callback=self.callback_face_landmarks,
            qos_profile=1
        )

        # state
        self.human = Human()
        self.controller = TabletController()

        # config
        self.debug = False

    # callbacks
    def callback_face_landmarks(self, msg):
        msg_data = msg.data.replace("\"", "")
        if msg_data == "None" or msg_data is None:
            return

        data = load_bad_json_data(msg_data)

        if data is None:
            return

        # update human estimate
        self.human.pose_estimate.set_face_estimate(data)

        # compute yaw action and  send command
        yaw_action = self.controller.get_tablet_yaw_action(self.human)
        move_msg = String()
        move_msg.data = str(json.dumps({'joint_wrist_yaw': yaw_action}))
        self.pub_tablet_move_by.publish(move_msg)

        # debug
        self.print(str(yaw_action))
        self.print(str(self.controller.get_head_vertical_vector(self.human)))
        self.print(str(self.controller.get_head_direction(self.human)))

    def print(self, msg):
        if self.debug:
            self.get_logger().info(str(msg))

    # main
    def main(self):
        rclpy.spin(self)

def main():
    rclpy.init()
    HeadTracker().main()
    rclpy.shutdown()
import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from .kinematics import load_bad_json_data
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
        # self.face_landmarks = None

    # callbacks
    def callback_face_landmarks(self, msg):
        msg_data = msg.data.replace("\"", "")
        if msg_data == "None" or msg_data is None:
            return

        data = load_bad_json_data(msg_data)

        if data is None:
            return

        # print(json.dumps(data, indent=2))
        # self.get_logger().info(str(data["chin_middle"]))
        # self.face_landmarks = data

        move_str = self.get_yaw_action(data["chin_middle"])
        self.get_logger().info(move_str)
        move_msg = String()
        move_msg.data = str(move_str)
        self.pub_tablet_move_by.publish(move_msg)

    # helpers
    def get_yaw_action(self, chin_xyz):
        x_deadband = [-0.1, 0.1]
        x = chin_xyz[0]
        if in_range(x, x_deadband):
            return json.dumps({"joint_wrist_yaw": 0.})
        
        Kp = 0.2
        yaw_action = Kp * (-1 * x)
        move_by_cmd = {
            "joint_wrist_yaw": yaw_action
        }
        return json.dumps(move_by_cmd)

    # main
    def main(self):
        rclpy.spin(self)

def main():
    rclpy.init()
    HeadTracker().main()
    rclpy.shutdown()
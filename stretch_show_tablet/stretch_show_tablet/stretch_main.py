import json

import numpy as np
import rclpy
from hello_helpers.hello_misc import HelloNode
from std_msgs.msg import String

PI_2 = np.pi / 2.0


def enforce_limits(value, min_value, max_value):
    return min([max([min_value, value]), max_value])


def enforce_joint_limits(pose: dict) -> dict:
    pose["lift"] = enforce_limits(pose["lift"], 0.25, 1.1)
    pose["arm_extension"] = enforce_limits(pose["arm_extension"], 0.02, 0.45)
    pose["yaw"] = enforce_limits(pose["yaw"], -PI_2, PI_2)
    pose["pitch"] = enforce_limits(pose["pitch"], -PI_2, PI_2)
    pose["roll"] = enforce_limits(pose["roll"], -PI_2, PI_2)

    return pose


class StretchMain(HelloNode):
    def __init__(self):
        super().__init__()
        self.main(node_name="stretch_main", node_topic_namespace="test_namespace")

        # sub
        self.create_subscription(
            String,
            "/stretch_show_tablet/move_by",
            callback=self.move_by,
            qos_profile=1,
        )

        self.create_subscription(
            String,
            "/stretch_show_tablet/goal",
            callback=self.move_to_goal,
            qos_profile=1,
        )

        self.init()
        rclpy.spin(self)

    def init(self):
        """
        other init
        """
        self.move_to_pose({"joint_wrist_roll": 0.0})

    # callbacks
    def move_by(self, msg):
        self.get_logger().info(str(msg.data))
        data = json.loads(msg.data)
        delta = data["joint_wrist_yaw"]
        if abs(delta) < 0.001:
            return

        max_delta = 0.1
        if abs(delta) > max_delta:
            delta = np.sign(delta) * max_delta

        # self.move
        current_state = self.joint_state
        names = current_state.name
        positions = current_state.position
        # self.get_logger().info(current_pose)
        current_yaw_position = positions[names.index("joint_wrist_yaw")]
        cmd_yaw_position = current_yaw_position + delta

        self.move_to_pose({"joint_wrist_yaw": cmd_yaw_position}, blocking=False)

    def move_to_goal(self, msg, confirm: bool = True):
        # msg_data = msg.data.replace("\"", "")
        data = json.loads(msg.data)
        pose = enforce_joint_limits(data)

        pose_cmd = {
            "rotate_mobile_base": pose["base"],
            "joint_lift": pose["lift"],
            "wrist_extension": pose["arm_extension"],
            "joint_wrist_yaw": pose["yaw"],
            "joint_wrist_pitch": pose["pitch"],
            "joint_wrist_roll": pose["roll"],
        }

        if confirm:
            print("moving to pose:")
            print(pose_cmd)
            if input("enter y to continue: ").lower() == "y":
                self.move_to_pose(pose_cmd)
            else:
                print("aborting!")
        else:
            self.move_to_pose(pose_cmd)


def main():
    StretchMain()


if __name__ == "__main__":
    main()

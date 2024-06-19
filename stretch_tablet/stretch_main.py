import rclpy
from hello_helpers.hello_misc import HelloNode
from std_msgs.msg import String
import json
import numpy as np

PI_2 = np.pi/2.

def enforce_limits(value, min_value, max_value):
    return min([max([min_value, value]), max_value])

def enforce_joint_limits(pose: dict) -> dict:
    pose["lift"] = enforce_limits(pose["lift"], 0.25, 1.1)
    pose["arm_extension"] = enforce_limits(pose["arm_extension"], 0.02, 0.45)
    pose["yaw"] = enforce_limits(pose["yaw"], -PI_2, PI_2)
    pose["pitch"] = enforce_limits(pose["yaw"], -PI_2, PI_2)
    pose["roll"] = enforce_limits(pose["yaw"], -PI_2, PI_2)

    return pose

class StretchMain(HelloNode):
    def __init__(self):
        super().__init__()
        self.main(node_name="stretch_main", node_topic_namespace="test_namespace")

        # sub
        self.create_subscription(
            String,
            "/stretch_tablet/goal",
            callback=self.move_to_goal,
            qos_profile=1
        )

        rclpy.spin(self)

    def move_to_goal(self, msg):
        # msg_data = msg.data.replace("\"", "")
        data = json.loads(msg.data)
        pose = enforce_joint_limits(data)

        pose_cmd = {
            "translate_mobile_base": pose["base"],
            "joint_lift": pose["lift"],
            "wrist_extension": pose["arm_extension"],
            "joint_wrist_yaw": pose["yaw"],
            "joint_wrist_pitch": pose["pitch"],
            "joint_wrist_roll": pose["roll"],
        }

        print("moving to pose:")
        print(pose_cmd)
        if input("enter y to continue: ").lower() == 'y':
            self.move_to_pose(pose_cmd)
        else:
            print("aborting!")

def main():
    StretchMain()

if __name__ == '__main__':
    main()
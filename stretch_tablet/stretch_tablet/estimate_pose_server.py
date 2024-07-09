import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String

# import sophuspy as sp
# import numpy as np
# from scipy.spatial.transform import Rotation as R

from stretch_tablet_interfaces.action import EstimateHumanPose

from enum import Enum
import json

from stretch_tablet.utils import load_bad_json_data
from stretch_tablet.human import Human

class EstimatePoseState(Enum):
    IDLE = 0
    ESTIMATE = 1
    DONE = 2
    ERROR = 99

class EstimatePoseActionServer(Node):
    def __init__(self):
        super().__init__("estimate_pose_action_server")
        self._action_server = ActionServer(
            self,
            EstimateHumanPose,
            'estimate_human_pose',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            cancel_callback=self.callback_action_cancel
        )

        # sub
        self.sub_body_landmarks = self.create_subscription(
            String,
            "/body_landmarks/landmarks_3d",
            callback=self.callback_body_landmarks,
            qos_profile=1
        )

        self._latest_human = Human()

    # get / set
    def get_latest_human(self):
        return self._latest_human

    # callbacks
    def callback_body_landmarks(self, msg: String):
        msg_data = msg.data.replace("\"", "")
        if msg_data == "None" or msg_data is None:
            return

        data = load_bad_json_data(msg_data)

        if data is None or data == "{}":
            return
        self._latest_human.pose_estimate.set_body_estimate(data)

    def callback_action_cancel(self, goal_handle):
        self.cleanup()
        return CancelResponse.ACCEPT
    
    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing Estimate Pose...')
        self.goal_handle = goal_handle
        n_samples = goal_handle.request.number_of_samples

        # init result
        result = EstimateHumanPose.Result()
        
        human = self.observe_human(n=n_samples)
        self.get_logger().info(str(json.dumps(human.pose_estimate.body_estimate)))

        result.body_pose_estimate = json.dumps(human.pose_estimate.body_estimate)
        return result

    # helpers
    def cleanup(self):
        pass

    def clear_last_pose_estimate(self):
        self._latest_human.pose_estimate.clear_estimates()

    def observe_human(self, n=1):
        human = Human()
        necessary_keys = [
            "nose",
            "neck",
            "right_shoulder",
            "left_shoulder"
        ]

        i = 0
        while i < n:
            latest_human = self.get_latest_human()
            if latest_human.pose_estimate.body_estimate is None:
                continue

            # check keys
            pose_keys = latest_human.pose_estimate.body_estimate.keys()
            for key in necessary_keys:
                if key not in pose_keys:
                    continue

            # TODO: probabilistic version averaging
            human.pose_estimate.set_body_estimate(latest_human.pose_estimate.body_estimate)
            self.clear_last_pose_estimate()
            i = i + 1

        return human

    # main
    def main(self):
        executor = MultiThreadedExecutor()

        action_server = EstimatePoseActionServer()

        rclpy.spin(action_server, executor=executor)

        action_server.destroy()

def main():
    rclpy.init()
    EstimatePoseActionServer().main()

if __name__ == '__main__':
    main()
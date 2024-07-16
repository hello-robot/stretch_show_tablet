import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action.server import ServerGoalHandle

from std_msgs.msg import String

import numpy as np

from stretch_tablet_interfaces.action import EstimateHumanPose

from enum import Enum
import json

from stretch_tablet.utils import load_bad_json_data
from stretch_tablet.human import HumanPoseEstimate

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

        self._latest_human_pose_estimate = HumanPoseEstimate()

    # get / set
    def get_latest_human_pose_estimate(self, filter_bad_points: bool=False):
        """
        Args:
            filter_bad_points (bool): whether to remove points without an
            accurate depth measurement
        """
        latest_human = self._latest_human_pose_estimate

        if latest_human is None or latest_human.body_estimate is None:
            return latest_human

        if filter_bad_points:
            # remove points that are adjacent to the camera
            filtered_human_estimate = {}
            for k in latest_human.body_estimate.keys():
                point = latest_human.body_estimate[k]
                if np.linalg.norm(point) > 0.02:
                    filtered_human_estimate[k] = point

            latest_human.set_body_estimate(filtered_human_estimate)

        return latest_human

    # callbacks
    def callback_body_landmarks(self, msg: String):
        msg_data = msg.data.replace("\"", "")
        if msg_data == "None" or msg_data is None:
            return

        data = load_bad_json_data(msg_data)

        if data is None or data == "{}":
            return
        self._latest_human_pose_estimate.set_body_estimate(data)

    def callback_action_cancel(self, goal_handle: ServerGoalHandle):
        self.cleanup()
        return CancelResponse.ACCEPT
    
    def execute_callback(self, goal_handle: ServerGoalHandle):
        self.get_logger().info('Executing Estimate Pose...')
        self.goal_handle = goal_handle
        n_samples = goal_handle.request.number_of_samples

        # ROS Action stuff
        result = EstimateHumanPose.Result()
        feedback = EstimateHumanPose.Feedback()
        
        pose_estimate = self.observe_human(n_samples=n_samples, goal_handle=goal_handle)
        if pose_estimate.body_estimate is not None and len([k for k in pose_estimate.body_estimate.keys()]) > 0:
            feedback.current_state = EstimatePoseState.DONE.value
            goal_handle.publish_feedback(feedback)
            goal_handle.succeed()
        else:
            goal_handle.abort()
            return result

        self.get_logger().info(str(json.dumps(pose_estimate.body_estimate)))

        result.body_pose_estimate = json.dumps(pose_estimate.body_estimate)
        return result

    # helpers
    def cleanup(self):
        pass

    def clear_last_pose_estimate(self):
        self._latest_human_pose_estimate.clear_estimates()

    def observe_human(self, n_samples: int=1, goal_handle:ServerGoalHandle=None) -> HumanPoseEstimate:
        """
        Args:
            n_samples (int): number of samples to average

        Returns:
            Human with populated pose estimate
        """
        if goal_handle is None:
            raise ValueError
        
        # ROS Action stuff
        feedback = EstimateHumanPose.Feedback()
        feedback.current_state = EstimatePoseState.ESTIMATE.value

        necessary_keys = [
            "nose",
            "neck",
            "right_shoulder",
            "left_shoulder"
        ]

        # loop inits
        rate = self.create_rate(10.)
        i = 0
        pose_estimates = [HumanPoseEstimate() for _ in range(n_samples)]

        while i < n_samples:
            feedback.number_of_samples_read = i
            goal_handle.publish_feedback(feedback)
            latest_pose = self.get_latest_human_pose_estimate(filter_bad_points=True)

            # check if human visible
            if latest_pose.body_estimate is None:
                rate.sleep()
                continue

            # get visible keypoints
            pose_keys = latest_pose.body_estimate.keys()

            # check if necessary keys visible
            can_see = True
            for key in necessary_keys:
                if key not in pose_keys:
                    self.get_logger().info("cannot see key joints!")
                    can_see = False
                    continue
            
            if not can_see:
                rate.sleep()
                continue

            pose_estimates[i].set_body_estimate(latest_pose.body_estimate)
            self.clear_last_pose_estimate()
            i = i + 1
            rate.sleep()

        # compute average estimate
        average_pose_estimate = HumanPoseEstimate.average_pose_estimates(pose_estimates)
        return average_pose_estimate

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
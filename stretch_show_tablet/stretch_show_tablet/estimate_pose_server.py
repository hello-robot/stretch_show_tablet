# mypy: ignore-errors
import json
from enum import Enum

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.action import ActionServer, CancelResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from stretch_show_tablet.human import HumanPoseEstimate
from stretch_show_tablet.utils import load_bad_json_data
from stretch_show_tablet_interfaces.action import EstimateHumanPose


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
            "estimate_human_pose",
            self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            cancel_callback=self.callback_action_cancel,
        )

        # sub
        self.sub_body_landmarks = self.create_subscription(
            String,
            "/body_landmarks/landmarks_3d",
            callback=self.callback_body_landmarks,
            qos_profile=1,
        )

        # tf
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(buffer=self.tf_buffer, node=self)

        self._latest_human_pose_estimate = HumanPoseEstimate()

    # get / set
    def get_latest_human_pose_estimate(self, filter_bad_points: bool = False):
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
                try:
                    point = latest_human.body_estimate[k]
                    if np.linalg.norm(point) > 0.02:
                        filtered_human_estimate[k] = point
                except KeyError:
                    pass

            latest_human.set_body_estimate(filtered_human_estimate)

        return latest_human

    # callbacks
    def callback_body_landmarks(self, msg: String):
        msg_data = msg.data.replace('"', "")
        if msg_data == "None" or msg_data is None:
            return

        data = load_bad_json_data(msg_data)

        if data is None or data == "{}":
            return
        self._latest_human_pose_estimate.set_body_estimate(data)

    def callback_action_cancel(self, goal_handle: ServerGoalHandle):
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle: ServerGoalHandle):
        self.get_logger().info("Executing Estimate Pose...")
        n_samples = goal_handle.request.number_of_samples

        # ROS Action stuff
        result = EstimateHumanPose.Result()
        feedback = EstimateHumanPose.Feedback()

        pose_estimate = self.observe_human(n_samples=n_samples, goal_handle=goal_handle)
        if (
            pose_estimate.body_estimate is not None
            and len([k for k in pose_estimate.body_estimate.keys()]) > 0
        ):
            feedback.current_state = EstimatePoseState.DONE.value
            goal_handle.publish_feedback(feedback)
            goal_handle.succeed()
        else:
            goal_handle.abort()
            return result

        self.get_logger().info(str(json.dumps(pose_estimate.body_estimate)))

        # construct result
        result.body_pose_estimate = json.dumps(pose_estimate.body_estimate)
        result.camera_pose_world = self.lookup_camera_pose()
        return result

    # helpers
    def cleanup(self):
        pass

    def clear_last_pose_estimate(self):
        self._latest_human_pose_estimate.clear_estimates()

    def lookup_camera_pose(self) -> PoseStamped:
        msg = PoseStamped()

        try:
            t = self.tf_buffer.lookup_transform(
                # "odom",
                "base_link",
                "camera_color_optical_frame",
                rclpy.time.Time(),
            )

            pos = Point(
                x=t.transform.translation.x,
                y=t.transform.translation.y,
                z=t.transform.translation.z,
            )
            msg.pose.position = pos
            msg.pose.orientation = t.transform.rotation
            msg.header.stamp = t.header.stamp

        except Exception as e:
            self.get_logger().error(str(e))
            self.get_logger().error(
                "EstimatePoseActionServer::lookup_camera_pose: returning empty pose!"
            )

        return msg

    def observe_human(
        self, n_samples: int = 1, goal_handle: ServerGoalHandle = None
    ) -> HumanPoseEstimate:
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

        necessary_keys = ["nose", "neck", "right_shoulder", "left_shoulder"]

        # loop inits
        rate = self.create_rate(10.0)
        i = 0
        pose_estimates = [HumanPoseEstimate() for _ in range(n_samples)]

        while i < n_samples and rclpy.ok():
            if goal_handle.is_cancel_requested:
                self.cleanup()
                goal_handle.canceled()
                break

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

        populated = i >= n_samples

        # compute average estimate
        if populated:
            average_pose_estimate = HumanPoseEstimate.average_pose_estimates(
                pose_estimates
            )
            return average_pose_estimate
        else:
            return HumanPoseEstimate()

    # main
    def main(self):
        executor = MultiThreadedExecutor()
        rclpy.spin(self, executor=executor)


def main():
    rclpy.init()
    EstimatePoseActionServer().main()


if __name__ == "__main__":
    main()

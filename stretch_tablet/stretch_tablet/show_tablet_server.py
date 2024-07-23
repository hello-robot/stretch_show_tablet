#!/usr/bin/env python3

# Standard Imports
import json
import sys
import threading
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

# Third-party imports
import rclpy
import sophuspy as sp
import tf2_py as tf2
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point, PoseStamped
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from stretch_tablet.human import Human, HumanPoseEstimate, transform_estimate_dict
from stretch_tablet.utils_ros import posestamped2se3
from stretch_tablet_interfaces.action import ShowTablet

# Local imports
from stretch_tablet_interfaces.srv import PlanTabletPose
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class ShowTabletActionServer(Node):
    """
    This node performs two functions:
     1. It estimates the human pose based on the `stretch_deep_perception` node,
        and publishes the output (e.g., for rendering to the web app). This
        functionality can be toggled on and off.
     2. It exposes an action server that uses the latest (non-stale) estimated
        human pose and moves the tablet to be in front of the user's face.
    """

    def __init__(
        self,
        # NOTE: all of the below can be made ROS2 parameters instead, if we
        # anticipate the client needing to change them during runtime.
        action_server_timeout_secs: float = 30.0,
        pose_history_window_size_n: int = 10,
        pose_history_window_size_secs: float = 2.0,
    ):
        """
        Initialize the node and its ROS interfaces.

        Parameters
        ----------
        action_server_timeout_secs : float
            The maximum time (in seconds) that the action server will wait for
            a goal to be complete before timing out.
        pose_history_window_size_n : int
            The number of poses to keep in the pose history.
        pose_history_window_size_secs : float
            The maximum time (in seconds) that a pose is considered "fresh" and
            not stale.
        """
        super().__init__("show_tablet_action_server")

        # Initialize the TF buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(buffer=self.tf_buffer, node=self)

        # Track the pose history
        self.pose_history_window_size_n = pose_history_window_size_n
        self.pose_history_window_size_secs = pose_history_window_size_secs
        self.pose_history_lock = threading.Lock()
        self.pose_history: List[Tuple[HumanPoseEstimate, Time]] = []

        # Create service and action clients for the functionality this node calls.
        self.plan_tablet_pose_srv = self.create_client(
            PlanTabletPose,
            "/plan_tablet_pose",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.toggle_perception_model_srv = self.create_client(
            SetBool,
            "/body_landmarks/detection/toggle",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.switch_to_position_mode_srv = self.create_client(
            Trigger,
            "/switch_to_position_mode",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.move_arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Create the publisher and subscriber for the human pose estimation, and
        # a service for toggling on/off human pose estimation.
        self.body_landmarks_pub = self.create_publisher(
            String,
            "/human_estimates/latest_body_pose",
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.body_landmarks_sub = self.create_subscription(
            String,
            "/body_landmarks/landmarks_3d",
            callback=partial(
                self.body_landmarks_callback, timeout=Duration(seconds=0.5)
            ),
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.pose_estimator_enabled_lock = threading.Lock()
        self.pose_estimator_enabled = False
        self.toggle_pose_estimation_srv = self.create_service(
            SetBool,
            "/toggle_body_pose_estimator",
            partial(
                self.toggle_pose_estimation_callback, timeout=Duration(seconds=2.0)
            ),
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Create the action server, and track the active goal request.
        self.active_goal_request_lock = threading.Lock()
        self.active_goal_request = None
        # self.action_server = ActionServer(
        #     self,
        #     ShowTablet,
        #     'show_tablet',
        #     partial(self.execute_callback, timeout=Duration(seconds=action_server_timeout_secs)),
        #     callback_group=ReentrantCallbackGroup(),
        #     cancel_callback=self.callback_action_cancel,
        #     goal_callback=self.callback_action_goal,
        # )

    def prune_pose_history(self, acquire_lock: bool = True) -> None:
        """
        Prune the pose history to keep only the most recent poses. Note: the
        pose history lock must be acquired before calling this function.

        Parameters
        ----------
        acquire_lock : bool
            Whether to acquire the pose history lock before pruning the pose history.
        """
        if acquire_lock:
            self.pose_history_lock.acquire()
        # Prune the pose history to keep the `pose_history_window_size_n` most recent poses
        # within the `pose_history_window_size_secs` time window.
        while len(self.pose_history) > self.pose_history_window_size_n:
            self.pose_history.pop(0)
        while len(self.pose_history) > 0 and (
            self.get_clock().now() - self.pose_history[0][1]
        ) > Duration(seconds=self.pose_history_window_size_secs):
            self.pose_history.pop(0)
        if acquire_lock:
            self.pose_history_lock.release()

    def push_to_pose_history(
        self, pose: HumanPoseEstimate, timestamp: Time, acquire_lock: bool = True
    ) -> None:
        """
        Push a pose to the pose history. Note: the pose history lock must be
        acquired before calling this function.

        Parameters
        ----------
        pose : HumanPoseEstimate
            The pose to push to the pose history.
        timestamp : Time
            The timestamp of the pose.
        acquire_lock : bool
            Whether to acquire the pose history lock before pushing to the pose history.
        """
        if acquire_lock:
            self.pose_history_lock.acquire()
        self.pose_history.append((pose, timestamp))
        self.prune_pose_history(acquire_lock=False)
        if acquire_lock:
            self.pose_history_lock.release()

    def get_poses_from_pose_history(
        self, acquire_lock: bool = True
    ) -> List[HumanPoseEstimate]:
        """
        Get the poses from the pose history. Note: the pose history lock must be
        acquired before calling this function.

        Parameters
        ----------
        acquire_lock : bool
            Whether to acquire the pose history lock before getting poses from the pose history.

        Returns
        -------
        List[HumanPoseEstimate]
            The poses from the pose history.
        """
        if acquire_lock:
            self.pose_history_lock.acquire()
        self.prune_pose_history(acquire_lock=False)
        retval = [pose for pose, _ in self.pose_history]
        if acquire_lock:
            self.pose_history_lock.release()
        return retval

    def __wait_for_future(
        self,
        future: Future,
        timeout: Duration,
        rate_hz: float = 10.0,
        check_termination: Optional[Callable] = None,
    ) -> bool:
        """
        Wait for a future to complete.

        Parameters
        ----------
        future : Future
            The future to wait for.
        timeout : Duration
            The maximum time to wait for the future to complete.
        rate_hz : float
            The rate at which to check if the future is done.
        check_termination : Optional[Callable]
            A function that returns True if we should terminate early, and False otherwise.

        Returns
        -------
        bool: True if the future completed, False otherwise.
        """
        start_time = self.get_clock().now()
        rate = self.create_rate(rate_hz)
        while rclpy.ok() and not future.done():
            # Check if the goal has been canceled
            if check_termination is not None and check_termination():
                self.get_logger().error("Goal was canceled!")
                return False

            # Check timeout
            if timeout is not None and (self.get_clock().now() - start_time) > timeout:
                self.get_logger().error("Timeout exceeded!")
                return False

            self.get_logger().debug("Waiting for future...", throttle_duration_sec=1.0)
            rate.sleep()
        return future.done()

    def toggle_pose_estimation_callback(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
        timeout: Duration,
    ) -> SetBool.Response:
        """
        Toggles the human pose estimation on or off, depending on the request.

        Parameters
        ----------
        request : SetBool.Request
            The request to toggle the human pose estimation on or off.
        response : SetBool.Response
            The response to the request.
        timeout : Duration
            The maximum time to wait for the service to respond.

        Returns
        -------
        SetBool.Response
            The response to the request.
        """
        self.get_logger().info(f"Received toggle request: {request.data}")
        start_time = self.get_clock().now()

        # Check if the body pose estimator service is available
        service_is_ready = self.toggle_perception_model_srv.wait_for_service(
            timeout_sec=remaining_time(self, start_time, timeout, return_secs=True)
        )
        if not service_is_ready:
            self.get_logger().error(
                "Toggle body pose estimation service is not available!"
            )
            response.success = False
            response.message = "Toggle body pose estimation service is not available"
            return response

        # Toggle on/off the body pose estimator service
        future = self.toggle_perception_model_srv.call_async(
            SetBool.Request(data=request.data)
        )
        future_finished = self.__wait_for_future(
            future, remaining_time(self, start_time, timeout)
        )
        if not future_finished:
            self.get_logger().error("Toggle service call failed or timed out!")
            response.success = False
            response.message = (
                "Toggle body pose estimation service call failed or timed out"
            )
            return response

        # Verify the response
        toggle_service_response = future.result()
        if not toggle_service_response.success:
            self.get_logger().error("Toggle service call was not successful!")
            response.success = False
            response.message = (
                "Toggle body pose estimation service call failed or timed out"
            )
            return response

        # Enable or disable the body pose estimator
        with self.pose_estimator_enabled_lock:
            self.pose_estimator_enabled = request.data

        # Return success
        response.success = True
        response.message = "Success"
        return response

    def body_landmarks_callback(self, msg: String, timeout: Duration) -> None:
        """
        Callback for the body landmarks subscriber. Process the body landmarks
        and publish the detected human pose (e.g., the one the action would use if
        it were invoked now).

        Parameters
        ----------
        msg : String
            The message containing the body landmarks.

        Returns
        -------
        None
        """
        self.get_logger().debug("Received body landmarks...", throttle_duration_sec=1.0)
        recv_time = self.get_clock().now()

        # Check if the pose estimator is enabled.
        with self.pose_estimator_enabled_lock:
            pose_estimator_enabled = self.pose_estimator_enabled
        if not pose_estimator_enabled:
            return

        # To be used, the JSON data must contain the following joints:
        necessary_joints = ["nose", "neck", "right_shoulder", "left_shoulder"]

        # Load the JSON data
        if len(msg.data) == 0:
            return
        try:
            human_keypoints_in_camera_frame = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(
                f"Error decoding JSON data from body landmarks: '{msg.data}'"
            )
            return
        for joint_name in necessary_joints:
            if joint_name not in human_keypoints_in_camera_frame:
                self.get_logger().error(
                    f"Missing joint '{joint_name}' in body landmarks!",
                    throttle_duration_sec=1.0,
                )
                return

        # Compute the pose estimate
        camera_pose_in_base_frame = self.lookup_camera_pose(
            timeout=remaining_time(self, recv_time, timeout),
        )
        pose_estimate = HumanPoseEstimate()
        pose_estimate.set_body_estimate_camera_frame(
            human_keypoints_in_camera_frame,
            camera_pose_in_base_frame,
        )

        # Add the pose estimate to the pose history
        self.push_to_pose_history(pose_estimate, recv_time)

        # Average the pose estimates and publish the result
        self.latest_average_pose = HumanPoseEstimate.average_pose_estimates(
            self.get_poses_from_pose_history()
        )
        latest_average_keypoints_robot_frame = (
            self.latest_average_pose.get_body_estimate_robot_frame()
        )
        latest_average_keypoints_camera_frame = transform_estimate_dict(
            latest_average_keypoints_robot_frame,
            camera_pose_in_base_frame.inverse(),
        )
        latest_average_keypoints_camera_frame = {
            k: v.tolist() for k, v in latest_average_keypoints_camera_frame.items()
        }
        self.body_landmarks_pub.publish(
            String(data=json.dumps(latest_average_keypoints_camera_frame))
        )

    def lookup_camera_pose(self, timeout: Duration) -> Optional[sp.SE3]:
        """
        Get the transform from the camera to the base frame.

        Parameters
        ----------
        timeout : Duration
            The maximum time to wait for the transform.

        Returns
        -------
        Optional[sp.SE3]
            The transform from the camera to the base frame, or None if the transform
            could not be found.
        """
        msg = PoseStamped()

        try:
            t = self.tf_buffer.lookup_transform(
                "base_link",
                "camera_color_optical_frame",
                rclpy.time.Time(),  # Use the latest transform
                timeout,
            )

            pos = Point(
                x=t.transform.translation.x,
                y=t.transform.translation.y,
                z=t.transform.translation.z,
            )
            msg.pose.position = pos
            msg.pose.orientation = t.transform.rotation
            msg.header.stamp = t.header.stamp

            return posestamped2se3(msg)

        except (
            tf2.ConnectivityException,
            tf2.ExtrapolationException,
            tf2.InvalidArgumentException,
            tf2.LookupException,
            tf2.TimeoutException,
            tf2.TransformException,
        ) as e:
            self.get_logger().error(
                f"Failed to get camera pose in base frame: {e}",
            )

        return None


def remaining_time(
    node: Node,
    start_time: Time,
    timeout: Duration,
    return_secs: bool = False,
) -> Union[Duration, float]:
    """
    Returns the remaining time until the timeout is reached.

    NOTE: If this code gets merged into `stretch_web_teleop`, this function
    should get consolidated with the analogous helper function in that repo.

    Parameters
    ----------
    node : Node
        The ROS node.
    start_time : Time
        The start time.
    timeout : Duration
        The timeout duration.
    return_secs : bool
        Whether to return the remaining time in seconds (float) or as a Duration.

    Returns
    -------
    Union[Duration, float]
        The remaining time until the timeout is reached.
    """
    elapsed_time = node.get_clock().now() - start_time
    remaining_time = Duration(
        nanoseconds=timeout.nanoseconds - elapsed_time.nanoseconds
    )
    return (remaining_time.nanoseconds / 1.0e9) if return_secs else remaining_time


def main(args: Optional[List[str]] = None):
    """
    The main entrypoint for ROS2.
    """
    rclpy.init(args=args)

    show_tablet_action_server = ShowTabletActionServer()
    show_tablet_action_server.get_logger().info("ShowTabletActionServer created!")

    # Use a MultiThreadedExecutor so that subscriptions, actions, etc. can be
    # processed in parallel.
    executor = MultiThreadedExecutor()
    rclpy.spin(show_tablet_action_server, executor=executor)

    show_tablet_action_server.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])

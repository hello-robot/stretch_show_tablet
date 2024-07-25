#!/usr/bin/env python3
from __future__ import annotations

# Standard Imports
import json
import sys
import threading
import traceback
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import rclpy
import sophuspy as sp
import tf2_py as tf2
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point, PoseStamped
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.client import Client
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from stretch_tablet.human import HumanPoseEstimate, transform_estimate_dict
from stretch_tablet.planner_helpers import JOINT_NAME_SHORT_TO_FULL, JOINT_LIMITS
from stretch_tablet.utils_ros import posestamped2se3
from stretch_tablet_interfaces.action import ShowTablet
from stretch_tablet_interfaces.srv import PlanTabletPose
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

class ShowTabletState(Enum):
    """
    The states of the ShowTablet action server.
    """

    ESTIMATE_HUMAN_POSE = 1
    PLAN_TABLET_POSE = 2
    NAVIGATE_BASE = 3  # not implemented yet
    MOVE_ARM_TO_TABLET_POSE = 4
    TERMINAL = 5
    ERROR = -1


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
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            callback=self.joint_state_callback,
            qos_profile=1,
            callback_group=MutuallyExclusiveCallbackGroup(),
            )
        self.joint_state = None
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
        self.human_pose_estimate_for_action: Optional[HumanPoseEstimate] = None
        self.robot_target_joint_positions_for_action: Optional[Dict[str, float]] = None
        self.action_server = ActionServer(
            self,
            ShowTablet,
            "show_tablet",
            partial(
                self.action_execute_callback,
                timeout=Duration(seconds=action_server_timeout_secs),
            ),
            goal_callback=self.action_goal_callback,
            cancel_callback=self.action_cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

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

    def joint_state_callback(self, msg: JointState) -> None:
        self.joint_state = msg

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
        if len(human_keypoints_in_camera_frame) == 0:
            return
        
        # filter points that are adjacent to camera (i.e., have bad color-depth alignment)
        filtered_human_estimate = {}
        for k in human_keypoints_in_camera_frame.keys():
            try:
                point = human_keypoints_in_camera_frame[k]
                if np.linalg.norm(point) > 0.02:
                    filtered_human_estimate[k] = point
            except KeyError:
                pass

        human_keypoints_in_camera_frame = filtered_human_estimate

        # check that all necessary joints are still present
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

        # Average the pose estimates
        latest_average_pose = HumanPoseEstimate.average_pose_estimates(
            self.get_poses_from_pose_history()
        )
        if len(latest_average_pose.get_body_estimate_robot_frame()) == 0:
            self.get_logger().error(
                "No valid body pose estimate found in the pose history!"
            )
            return
        latest_average_keypoints_robot_frame = (
            latest_average_pose.get_body_estimate_robot_frame()
        )

        # Convert the result from robot frame to camera frame, and publish
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

    def action_goal_callback(self, goal_request: ShowTablet.Goal) -> GoalResponse:
        """
        Accept a goal if this action does not already have an active goal, else reject.

        Parameters
        ----------
        goal_request : ShowTablet.Goal
            The goal request.

        Returns
        -------
        GoalResponse
            The response to the goal request.
        """
        self.get_logger().info(f"Received request {goal_request}")

        # Recompute the average pose (to remove stale poses), and reject the goal
        # if we don't have a valid pose estimate (which includes the case where
        # all poses are stale).
        latest_average_pose = HumanPoseEstimate.average_pose_estimates(
            self.get_poses_from_pose_history()
        )
        if (
            latest_average_pose is None
            or len(latest_average_pose.get_body_estimate_robot_frame()) == 0
        ):
            self.get_logger().info(
                "Rejecting goal request since there is no valid pose estimate. "
                "Be sure to toggle detection on and ensure that messages are being "
                "sent on the /human_estimates/latest_body_pose topic."
            )
            return GoalResponse.REJECT

        # Reject the goal is there is already an active goal
        with self.active_goal_request_lock:
            if self.active_goal_request is not None:
                self.get_logger().info(
                    "Rejecting goal request since there is already an active one"
                )
                return GoalResponse.REJECT

        # Accept the goal
        self.get_logger().info("Accepting goal request")
        self.active_goal_request = goal_request
        self.human_pose_estimate_for_action = None
        self.robot_target_joint_positions_for_action = None
        return GoalResponse.ACCEPT

    def action_cancel_callback(self, goal_handle: ServerGoalHandle) -> CancelResponse:
        """
        Always accept client requests to cancel the active goal.

        Parameters
        ----------
        goal_handle : ServerGoalHandle
            The goal handle.

        Returns
        -------
        CancelResponse
            The response to the cancel request.
        """
        self.get_logger().info("Received cancel request, accepting")
        return CancelResponse.ACCEPT

    def check_ok(
        self, goal_handle: ServerGoalHandle, start_time: Time, timeout: Duration
    ) -> bool:
        """
        Check if the goal is still OK to execute.

        Parameters
        ----------
        goal_handle : ServerGoalHandle
            The goal handle.
        start_time : Time
            The start time of the goal.
        timeout : Duration
            The maximum time to wait for the goal to complete.

        Returns
        -------
        bool
            True if the goal is still OK to execute, False otherwise.
        """
        # Check if the goal has been canceled
        if goal_handle.is_cancel_requested:
            self.get_logger().debug("Goal was canceled!")
            return False

        # Check timeout
        if (self.get_clock().now() - start_time) > timeout:
            self.get_logger().debug("Timeout exceeded!")
            return False

        return rclpy.ok()

    def cleanup_action_execution(self) -> None:
        """
        Cleanup after the goal has been completed or canceled.
        """
        # Clear the active goal request
        with self.active_goal_request_lock:
            self.active_goal_request = None

        # Clear the human pose estimate for the action
        self.human_pose_estimate_for_action = None

        # Switch robot back to navigation mode
        # self.

    def error_action_execution(
        self,
        goal_handle: ServerGoalHandle,
        error_msg: str,
    ) -> ShowTablet.Result:
        """
        Handle an error in the action server goal.

        Parameters
        ----------
        goal_handle : ServerGoalHandle
            The goal handle.
        error_msg : str
            The error message.

        Returns
        -------
        ShowTablet.Result
            The result of the action server goal.
        """
        self.get_logger().error(error_msg)
        self.cleanup_action_execution()
        goal_handle.abort()
        return ShowTablet.Result(status=ShowTablet.Result.STATUS_ERROR)

    def cancel_action_execution(
        self,
        goal_handle: ServerGoalHandle,
        cancel_msg: str,
    ) -> ShowTablet.Result:
        """
        Handle a cancel request in the action server goal.

        Parameters
        ----------
        goal_handle : ServerGoalHandle
            The goal handle.
        cancel_msg : str
            The cancel message.

        Returns
        -------
        ShowTablet.Result
            The result of the action server goal.
        """
        self.get_logger().info(cancel_msg)
        self.cleanup_action_execution()
        goal_handle.canceled()
        return ShowTablet.Result(status=ShowTablet.Result.STATUS_CANCELED)

    def succeed_action_execution(
        self,
        goal_handle: ServerGoalHandle,
        success_msg: str,
    ) -> ShowTablet.Result:
        """
        Handle a successful completion of the action server goal.

        Parameters
        ----------
        goal_handle : ServerGoalHandle
            The goal handle.
        success_msg : str
            The success message.

        Returns
        -------
        ShowTablet.Result
            The result of the action server goal.
        """
        self.get_logger().info(success_msg)
        self.cleanup_action_execution()
        goal_handle.succeed()
        return ShowTablet.Result(status=ShowTablet.Result.STATUS_SUCCESS)

    def action_execute_callback(
        self,
        goal_handle: ServerGoalHandle,
        timeout: Duration,
    ) -> ShowTablet.Result:
        """
        Execute the action server goal, by going through the state machine.

        Parameters
        ----------
        goal_handle : ServerGoalHandle
            The goal handle.
        timeout : Duration
            The maximum time to wait for the goal to complete.

        Returns
        -------
        ShowTablet.Result
            The result of the action server goal.
        """
        self.get_logger().info("Executing goal...")
        start_time = self.get_clock().now()

        # Run the state machine
        rate = self.create_rate(10.0)
        state = ShowTabletState.ESTIMATE_HUMAN_POSE
        state_generator = None
        while self.check_ok(goal_handle, start_time, timeout):
            # If we don't yet have a state generator, create one
            if state_generator is None:
                if state == ShowTabletState.ESTIMATE_HUMAN_POSE:
                    state_generator = self.execute_estimate_human_pose(
                        goal_handle, timeout
                    )
                elif state == ShowTabletState.PLAN_TABLET_POSE:
                    state_generator = self.execute_plan_tablet_pose(
                        goal_handle, timeout
                    )
                elif state == ShowTabletState.NAVIGATE_BASE:
                    state_generator = self.execute_navigate_base(goal_handle, timeout)
                elif state == ShowTabletState.MOVE_ARM_TO_TABLET_POSE:
                    state_generator = self.execute_move_arm_to_tablet_pose(
                        goal_handle, timeout
                    )
                elif state == ShowTabletState.TERMINAL:
                    return self.succeed_action_execution(
                        goal_handle,
                        "Goal completed successfully!",
                    )
                elif state == ShowTabletState.ERROR:
                    break
                else:
                    self.get_logger().error(f"Invalid state: {state}")
                    return self.error_action_execution(
                        goal_handle,
                        "Invalid state!",
                    )
            # If we have a state generator, execute it until it yields a new state
            else:
                try:
                    old_state = state
                    state = next(state_generator)
                    if state != old_state:
                        state_generator = None
                except Exception as e:
                    self.get_logger().error(traceback.format_exc())
                    return self.error_action_execution(
                        goal_handle,
                        f"Error executing state machine: {e}",
                    )
            rate.sleep()

        # If we get here, the goal either encountered an error, timed out,
        # or was canceled. If there is a state_generator, run it one more
        # time to properly clean up before returning.
        if state_generator is not None:
            try:
                state = next(state_generator)
            except StopIteration:
                pass
            except Exception as e:
                self.get_logger().error(traceback.format_exc())
                return self.error_action_execution(
                    goal_handle,
                    f"Error cleaning up generator for state {state}: {e}",
                )
        if goal_handle.is_cancel_requested:
            return self.cancel_action_execution(
                goal_handle,
                "Goal was canceled. Has exited with cleanup.",
            )
        return self.error_action_execution(
            goal_handle,
            "Goal execution encountered an error or timed out. Has exited with cleanup.",
        )

    def execute_estimate_human_pose(
        self,
        goal_handle: ServerGoalHandle,
        timeout: Duration,
    ) -> Generator[ShowTabletState, None, None]:
        """
        Execute the ESTIMATE_HUMAN_POSE state.

        Parameters
        ----------
        goal_handle : ServerGoalHandle
            The goal handle.
        timeout : Duration
            The maximum time to wait for the state to complete.

        Yields
        ------
        ShowTabletState
            Whether to continue executing this state (if it returns the same state),
            or move to a new state (if it returns a different state).
        """
        # Recompute the average pose (to remove stale poses).
        latest_average_pose = HumanPoseEstimate.average_pose_estimates(
            self.get_poses_from_pose_history()
        )
        if (
            latest_average_pose is None
            or len(latest_average_pose.get_body_estimate_robot_frame()) == 0
        ):
            self.get_logger().error(
                "No valid body pose estimate found in the pose history!"
            )
            yield ShowTabletState.ERROR
            return

        # Store the latest average pose estimate. At this point, it remains fixed
        # until the goal terminates.
        self.human_pose_estimate_for_action = latest_average_pose
        yield ShowTabletState.PLAN_TABLET_POSE
        return

    def call_service(
        self,
        goal_handle: ServerGoalHandle,
        timeout: Duration,
        service_client: Client,
        request: Any,
        curr_state: ShowTabletState,
    ) -> Generator[ShowTabletState, None, Any]:
        """
        A helper function that checks whether a service is ready, calls the service,
        and returns the response.

        Parameters
        ----------
        goal_handle : ServerGoalHandle
            The goal handle.
        timeout : Duration
            The maximum time to wait for the service to complete.
        service_client : Client
            The service client.
        request : Any
            The request to the service.
        curr_state : ShowTabletState
            The current state of the action server.

        Yields
        ------
        ShowTabletState
            Whether to continue executing this state (if it returns the same state),
            or move to a new state (if it returns a different state).

        Returns
        -------
        Any
            The response from the service, or None if the service call failed.
        """
        start_time = self.get_clock().now()

        # Check if the service is available
        service_is_ready = False
        while self.check_ok(goal_handle, start_time, timeout):
            service_is_ready = service_client.service_is_ready()
            if service_is_ready:
                break
            yield curr_state
        if not service_is_ready:
            self.get_logger().error(
                f"{service_client.srv_name} service is not available!"
            )
            yield ShowTabletState.ERROR
            return

        # Call the service
        future = service_client.call_async(request)
        done = False
        while self.check_ok(goal_handle, start_time, timeout) and not done:
            done = future.done()
            yield curr_state
        if not done:
            self.get_logger().error(
                f"{service_client.srv_name} service call timed out!"
            )
            service_client.remove_pending_request(future)
            yield ShowTabletState.ERROR
            return
        response = future.result()
        self.get_logger().info(
            f"{service_client.srv_name} service response: {response}"
        )
        return response

    def call_action(
        self,
        goal_handle: ServerGoalHandle,
        timeout: Duration,
        action_client: ActionClient,
        goal: Any,
        curr_state: ShowTabletState,
    ) -> Generator[ShowTabletState, None, Any]:
        """
        A helper function that checks whether an action is ready, calls the action,
        and returns the result.

        Parameters
        ----------
        goal_handle : ServerGoalHandle
            The goal handle.
        timeout : Duration
            The maximum time to wait for the action to complete.
        action_client : ActionClient
            The action client.
        goal : Any
            The goal to the action.
        curr_state : ShowTabletState
            The current state of the action server.

        Yields
        ------
        ShowTabletState
            Whether to continue executing this state (if it returns the same state),
            or move to a new state (if it returns a different state).

        Returns
        -------
        Any
            The result from the action, or None if the action call failed.
        """
        start_time = self.get_clock().now()

        # Check if the action is available
        action_is_ready = False
        while self.check_ok(goal_handle, start_time, timeout):
            action_is_ready = action_client.server_is_ready()
            if action_is_ready:
                break
            yield curr_state
        if not action_is_ready:
            self.get_logger().error(
                f"{action_client._action_name} action is not available!"
            )
            yield ShowTabletState.ERROR
            return

        # Send the goal to the action
        future = action_client.send_goal_async(goal)
        done = False
        while self.check_ok(goal_handle, start_time, timeout) and not done:
            done = future.done()
            yield curr_state
        if not done:
            self.get_logger().error(
                f"{action_client._action_name} action call timed out while waiting "
                "for the goal to be accepted/rejected"
            )
            yield ShowTabletState.ERROR
            return

        # Check if the goal was accepted
        client_goal_handle = future.result()
        if not client_goal_handle.accepted:
            self.get_logger().error(
                f"{action_client._action_name} action goal was rejected!"
            )
            yield ShowTabletState.ERROR
            return

        # Wait for the action to complete
        future = client_goal_handle.get_result_async()
        done = False
        while self.check_ok(goal_handle, start_time, timeout) and not done:
            self.get_logger().info("Waiting for action to complete...")
            done = future.done()
            yield curr_state
        self.get_logger().info(f"Action completed {done}!")
        if not done:
            self.get_logger().error(
                f"{action_client._action_name} action call timed out while waiting "
                "for the goal to complete"
            )
            # Send a cancellation request, but don't wait for it to complete
            _ = client_goal_handle.cancel_goal_async()
            yield ShowTabletState.ERROR
            return
        result = future.result()
        self.get_logger().info(f"{action_client._action_name} action result: {result}")
        return result

    def execute_plan_tablet_pose(
        self,
        goal_handle: ServerGoalHandle,
        timeout: Duration,
    ) -> Generator[ShowTabletState, None, None]:
        """
        Execute the PLAN_TABLET_POSE state.

        Parameters
        ----------
        goal_handle : ServerGoalHandle
            The goal handle.
        timeout : Duration
            The maximum time to wait for the state to complete.

        Yields
        ------
        ShowTabletState
            Whether to continue executing this state (if it returns the same state),
            or move to a new state (if it returns a different state).
        """
        start_time = self.get_clock().now()

        # Get the latest human pose estimate
        human_pose_estimate = self.human_pose_estimate_for_action
        if human_pose_estimate is None:
            self.get_logger().error("No human pose estimate found!")
            yield ShowTabletState.ERROR
            return

        # Call the plan tablet pose service
        service_generator = self.call_service(
            goal_handle,
            remaining_time(self, start_time, timeout),
            self.plan_tablet_pose_srv,
            PlanTabletPose.Request(
                human_joint_dict_robot_frame=human_pose_estimate.get_body_estimate_string(),
            ),
            ShowTabletState.PLAN_TABLET_POSE,
        )
        response = None
        try:
            while True:
                yield next(service_generator)
        except StopIteration as e:
            response = e.value
        if response is None or not response.success:
            self.get_logger().error("Plan tablet pose service call was not successful!")
            yield ShowTabletState.ERROR
            return

        # Get the planner result
        joint_names = response.robot_ik_joint_names
        joint_positions = response.robot_ik_joint_positions
        joint_dict = {n: p for n, p in zip(joint_names, joint_positions)}
        enforce_joint_limits(joint_dict)
        # Remove small motions, which the robot base cannot execute
        if abs(joint_dict["base"]) < 0.2:
            joint_dict.pop("base")
        self.get_logger().info(f"Planned joint positions: {joint_dict}")

        # Store the robot target joint positions for the action. At this point, it
        # remains fixed until the goal terminates.
        self.robot_target_joint_positions_for_action = joint_dict
        yield ShowTabletState.NAVIGATE_BASE
        return

    def execute_navigate_base(
        self,
        goal_handle: ServerGoalHandle,
        timeout: Duration,
    ) -> Generator[ShowTabletState, None, None]:
        """
        Execute the NAVIGATE_BASE state.

        Parameters
        ----------
        goal_handle : ServerGoalHandle
            The goal handle.
        timeout : Duration
            The maximum time to wait for the state to complete.

        Yields
        ------
        ShowTabletState
            Whether to continue executing this state (if it returns the same state),
            or move to a new state (if it returns a different state).
        """
        # Not yet implemented
        yield ShowTabletState.MOVE_ARM_TO_TABLET_POSE
        return

    def create_move_to_pose_goal(
        self,
        joint_dict: Dict[str, float],
        move_time_sec: float = 4.0,
    ) -> FollowJointTrajectory.Goal:
        """
        Create a FollowJointTrajectory goal to move the robot to the specified joint positions.

        Parameters
        ----------
        joint_dict : Dict[str, float]
            The joint positions to move the robot to.
        move_time_sec : float
            The time (in seconds) to move the robot to the specified joint positions.

        Returns
        -------
        FollowJointTrajectory.Goal
            The goal to move the robot to the specified joint positions.
        """
        joint_names = [k for k in joint_dict.keys()]
        joint_positions = [joint_dict[k] for k in joint_names]
        trajectory = FollowJointTrajectory.Goal()
        trajectory.trajectory.joint_names = [
            JOINT_NAME_SHORT_TO_FULL[j] for j in joint_names
        ]
        trajectory.trajectory.points = [JointTrajectoryPoint()]
        trajectory.trajectory.points[0].positions = [p for p in joint_positions]
        trajectory.trajectory.points[0].time_from_start = Duration(
            seconds=move_time_sec
        ).to_msg()
        return trajectory

    def execute_move_arm_to_tablet_pose(
        self,
        goal_handle: ServerGoalHandle,
        timeout: Duration,
    ) -> Generator[ShowTabletState, None, None]:
        """
        Execute the MOVE_ARM_TO_TABLET_POSE state.

        Parameters
        ----------
        goal_handle : ServerGoalHandle
            The goal handle.
        timeout : Duration
            The maximum time to wait for the state to complete.

        Yields
        ------
        ShowTabletState
            Whether to continue executing this state (if it returns the same state),
            or move to a new state (if it returns a different state).
        """
        start_time = self.get_clock().now()

        # Create the poses, which will be executed in sequence.
        pose_tuck = {"arm_extension": 0.05, "yaw": 0.0}
        if "base" in self.robot_target_joint_positions_for_action.keys():
            pose_base = {
                "base": self.robot_target_joint_positions_for_action["base"],
                "lift": self.robot_target_joint_positions_for_action["lift"],
            }
        else:
            pose_base = {
                "lift": self.robot_target_joint_positions_for_action["lift"]
                }
        pose_arm = {
            "arm_extension": self.robot_target_joint_positions_for_action[
                "arm_extension"
            ],
        }
        pose_wrist = {
            "yaw": self.robot_target_joint_positions_for_action["yaw"],
            "pitch": self.robot_target_joint_positions_for_action["pitch"],
            # "roll": self.robot_target_joint_positions_for_action["roll"],
            "roll": 0.,  # Keep the wrist roll at 0 for now
        }

        curr_head_pan = self.joint_state.position[self.joint_state.name.index("joint_head_pan")]

        if "base" in pose_base.keys():
            base_theta = pose_base["base"]
        else:
            base_theta = 0.
            
        target_head_pan = curr_head_pan - base_theta
        while target_head_pan < np.deg2rad(-230.):
            target_head_pan += 2.0 * np.pi
        while target_head_pan > np.deg2rad(90.):
            target_head_pan -= 2.0 * np.pi

        pose_head = {
            "head_pan": target_head_pan
        }
        poses = [pose_tuck, pose_base, pose_arm, pose_wrist, pose_head]

        # Call the switch to position mode service
        service_generator = self.call_service(
            goal_handle,
            remaining_time(self, start_time, timeout),
            self.switch_to_position_mode_srv,
            Trigger.Request(),
            ShowTabletState.MOVE_ARM_TO_TABLET_POSE,
        )
        response = None
        try:
            while True:
                yield next(service_generator)
        except StopIteration as e:
            response = e.value
        if response is None or not response.success:
            self.get_logger().error(
                "Switch to position mode service was not successful!"
            )
            yield ShowTabletState.ERROR
            return

        # Execute the pose motions
        for pose in poses:
            goal = self.create_move_to_pose_goal(pose)
            action_generator = self.call_action(
                goal_handle,
                remaining_time(self, start_time, timeout),
                self.move_arm_client,
                goal,
                ShowTabletState.MOVE_ARM_TO_TABLET_POSE,
            )
            result = None
            try:
                while True:
                    yield next(action_generator)
            except StopIteration as e:
                result = e.value
            if (
                result is None
                or result.status != GoalStatus.STATUS_SUCCEEDED
                or result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL
            ):
                self.get_logger().error(
                    "Move arm to tablet pose action was not successful!"
                )
                yield ShowTabletState.ERROR
                return

        # Return the terminal state
        yield ShowTabletState.TERMINAL
        return


def enforce_joint_limits(pose: Dict[str, float]) -> None:
    """
    Modifies the robot positions in `pose` by truncating them at the joint limits.

    Parameters
    ----------
    pose : Dict[str, float]
        The robot positions to enforce joint limits on.

    Returns
    -------
    None (this function destructively modifies the input dictionary).
    """
    for key in pose.keys():
        if key in JOINT_LIMITS.keys():
            pose[key] = np.clip(pose[key], JOINT_LIMITS[key][0], JOINT_LIMITS[key][1])

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

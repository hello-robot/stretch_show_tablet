import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.time import Time

from std_msgs.msg import String
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped, Point
from std_srvs.srv import SetBool, Trigger

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import sophuspy as sp
import numpy as np
from scipy.spatial.transform import Rotation as R

from stretch_tablet_interfaces.srv import PlanTabletPose
from stretch_tablet_interfaces.action import ShowTablet

from enum import Enum
import json
import threading
from typing import Callable, Optional

from stretch_tablet.utils import load_bad_json_data
from stretch_tablet.utils_ros import generate_pose_stamped, posestamped2se3
from stretch_tablet.human import Human, HumanPoseEstimate

# testing
import time

# helpers
JOINT_NAME_SHORT_TO_FULL = {
    "base": "rotate_mobile_base",
    "lift": "joint_lift",
    "arm_extension": "wrist_extension",
    "yaw": "joint_wrist_yaw",
    "pitch": "joint_wrist_pitch",
    "roll": "joint_wrist_roll",
}

JOINT_NAME_FULL_TO_SHORT = {v: k for k, v in JOINT_NAME_SHORT_TO_FULL.items()}

PI_2 = np.pi/2.

def enforce_limits(value, min_value, max_value):
    return min([max([min_value, value]), max_value])

def enforce_joint_limits(pose: dict) -> dict:
    pose["base"] = enforce_limits(pose["base"], -PI_2, PI_2)
    pose["lift"] = enforce_limits(pose["lift"], 0.25, 1.1)
    pose["arm_extension"] = enforce_limits(pose["arm_extension"], 0.02, 0.45)
    pose["yaw"] = enforce_limits(pose["yaw"], -np.deg2rad(60.), PI_2)
    pose["pitch"] = enforce_limits(pose["pitch"], -PI_2, 0.2)
    # pose["roll"] = enforce_limits(pose["roll"], -PI_2, PI_2)
    pose["roll"] = 0.

    return pose

# classes
class ShowTabletState(Enum):
    IDLE = 0
    ESTIMATE_HUMAN_POSE = 1
    PLAN_TABLET_POSE = 2
    NAVIGATE_BASE = 3  # not implemented yet
    MOVE_ARM_TO_TABLET_POSE = 4
    END_INTERACTION = 5
    EXIT = 98
    ABORT = 99
    ERROR = -1

# TODO: This node doesn't cleanly handle keyboard interrupts while the action is executing
# (I noticed it when it was waiting for the plan tablet pose service), but it should.

# TODO: I've implemented some cancellation logic, but it should be tested more thoroughly.
# Ideally, every function, loop, and blocking call should be checking for cancellations.
# And cancellations should be passed down to any other ROS services or actions that are
# being called.

class ShowTabletActionServer(Node):
    def __init__(self):
        super().__init__('show_tablet_action_server')

        # Store the latest average pose estimate.
        # TODO: Also store the timestamp this was computed at.
        self.latest_average_pose = None
        
        # pub
        self.pub_valid_estimates = self.create_publisher(
            String,
            "/human_estimates/latest_body_pose",
            qos_profile=1
        )
        
        # sub
        self.sub_body_landmarks = self.create_subscription(
            String,
            "/body_landmarks/landmarks_3d",
            callback=self.callback_body_landmarks,
            qos_profile=1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Toggle on/off person detection
        self.toggle_service_timeout = 2.0  # secs
        self.toggle_service = self.create_service(
            SetBool, "/toggle_body_pose_estimator", self.toggle_body_pose_estimator_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # srv clients
        self.srv_plan_tablet_pose = self.create_client(
            PlanTabletPose, 'plan_tablet_pose', callback_group=MutuallyExclusiveCallbackGroup()
        )
        
        self.srv_toggle_detection = self.create_client(
            SetBool, '/body_landmarks/detection/toggle', callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.srv_toggle_position_mode = self.create_client(
            Trigger, '/switch_to_position_mode', callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.srv_toggle_navigation_mode = self.create_client(
            Trigger, '/switch_to_navigation_mode', callback_group=MutuallyExclusiveCallbackGroup()
        )
        
        # motion
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # tf
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(buffer=self.tf_buffer, node=self)
        
        # state
        self.feedback = ShowTablet.Feedback()
        self.goal_handle = None
        self.human = Human()
        self.abort = False
        self.result = ShowTablet.Result()

        self._pose_history = []

        self._robot_joint_target = None
        self._pose_estimator_enabled_lock = threading.Lock()
        self._pose_estimator_enabled = False

        # config
        self._robot_move_time_s = 4.
        self._n_poses = 10

        # Create the shared resource to ensure that the action server rejects all
        # new goals while a goal is currently active.
        self.active_goal_request_lock = threading.Lock()
        self.active_goal_request = None

        # Create the action server
        self._action_server = ActionServer(
            self,
            ShowTablet,
            'show_tablet',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            cancel_callback=self.callback_action_cancel,
            goal_callback=self.callback_action_goal,
        )

    # helpers
    def now(self) -> Time:
        return self.get_clock().now().to_msg()
    
    def get_pose_history(self, max_age=float("inf")):
        return [p for (p, t) in self._pose_history]

    def build_tablet_pose_request(self):
        request = PlanTabletPose.Request()
        human = self.human

        if not self.human.pose_estimate.is_body_populated():
            self.get_logger().error("ShowTabletActionServer::build_tablet_pose_request: self.human empty!")
            return request

        # construct request
        request.human_joint_dict_robot_frame = human.pose_estimate.get_body_estimate_string()

        return request
    
    def lookup_camera_pose(self) -> PoseStamped:
        msg = PoseStamped()

        try:
            t = self.tf_buffer.lookup_transform(
                    # "odom",
                    "base_link",
                    "camera_color_optical_frame",
                    rclpy.time.Time())
            
            pos = Point(x=t.transform.translation.x,
                        y=t.transform.translation.y,
                        z=t.transform.translation.z)
            msg.pose.position = pos
            msg.pose.orientation = t.transform.rotation
            msg.header.stamp = t.header.stamp

        except Exception as e:
            self.get_logger().error(str(e))
            self.get_logger().error("EstimatePoseActionServer::lookup_camera_pose: returning empty pose!")
        
        return msg
    
    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def cleanup(self):
        # TODO: implement
        pass

    def __push_to_pose_history(self, pose_estimate: HumanPoseEstimate):
        entry = (pose_estimate, self.now())
        self._pose_history.append(entry)

        while len(self._pose_history) > self._n_poses:
            self._pose_history.pop(0)

    def __wait_for_future(
        self, 
        future, 
        rate_hz: float = 10.0,
        timeout: Optional[Duration] = None,
        check_termination: Optional[Callable] = None,
    ) -> bool:
        """
        Wait for a future to complete.

        Parameters
        ----------
        future: The future to wait for.
        rate_hz: The rate at which to check the future.
        timeout: The maximum time to wait for the future to complete.

        Returns
        -------
        bool: True if the future completed, False otherwise.
        """
        # TODO: The timeout should be required, not optional!
        start_time = self.get_clock().now()
        rate = self.create_rate(rate_hz)
        while rclpy.ok() and not future.done():
            # Check if the goal has been canceled
            if check_termination is not None and check_termination():
                return False

            # Check timeout
            if timeout is not None and (self.get_clock().now() - start_time) > timeout:
                self.get_logger().error("Timeout exceeded!")
                return False
            
            self.get_logger().debug("Waiting for future...", throttle_duration_sec=1.0)
            rate.sleep()
        return future.done()

    def __move_to_pose(self, joint_dict: dict, blocking: bool=True):
        joint_names = [k for k in joint_dict.keys()]
        joint_positions = [v for v in joint_dict.values()]

        # build message
        trajectory = FollowJointTrajectory.Goal()
        trajectory.trajectory.joint_names = [JOINT_NAME_SHORT_TO_FULL[j] for j in joint_names]
        trajectory.trajectory.points = [JointTrajectoryPoint()]
        trajectory.trajectory.points[0].positions = [p for p in joint_positions]
        trajectory.trajectory.points[0].time_from_start = Duration(seconds=self._robot_move_time_s).to_msg()

        future = self.arm_client.send_goal_async(trajectory)
        future_finished = self.__wait_for_future(
            future, 
            check_termination=lambda: self.goal_handle.is_cancel_requested,
        )   # TODO: add a timeout!
        if blocking and future_finished:
            goal_handle = future.result()
            get_result_future = goal_handle.get_result_async()
            future_finished = self.__wait_for_future(
                get_result_future, 
                check_termination=lambda: self.goal_handle.is_cancel_requested,
            )  # TODO: add a timeout!
            if not future_finished and self.goal_handle.is_cancel_requested:
                future = goal_handle.cancel_goal()
                self.__wait_for_future(future)  # TODO: add a timeout!

    def __present_tablet(self, joint_dict: dict):
        pose_tuck = {
            "arm_extension": 0.05,
            "yaw": 0.
        }

        if "base" in joint_dict.keys():
            pose_base = {
                "base": joint_dict["base"],
                "lift": joint_dict["lift"],
            }
        else:
            pose_base = {
                "lift": joint_dict["lift"]
            }

        pose_arm = {
            "arm_extension": joint_dict["arm_extension"],
        }

        pose_wrist = {
            "yaw": joint_dict["yaw"],
            "pitch": joint_dict["pitch"],
            "roll": joint_dict["roll"],
        }

        # sequence
        self.srv_toggle_position_mode.call(Trigger.Request())

        self.__move_to_pose(pose_tuck, blocking=True)
        self.__move_to_pose(pose_base, blocking=True)
        self.__move_to_pose(pose_arm, blocking=True)
        self.__move_to_pose(pose_wrist, blocking=True)
        
    # callbacks
    def toggle_body_pose_estimator_callback(
        self, 
        request: SetBool.Request, 
        response: SetBool.Response,
    ) -> SetBool.Response:
        self.get_logger().info(f"Received toggle request: {request.data}")
        start_time = self.get_clock().now()

        # Check if the body pose estimator service is available
        if not self.srv_toggle_detection.wait_for_service(timeout_sec=self.toggle_service_timeout):
            self.get_logger().error("Toggle service not available!")
            response.success = False
            response.message = "Toggle body pose estimation service not available"
            return response

        # Toggle on/off the body pose estimator service
        future = self.srv_toggle_detection.call_async(SetBool.Request(data=request.data))
        future_finished = self.__wait_for_future(future, timeout=(self.get_clock().now() - start_time))
        if not future_finished:
            self.get_logger().error("Toggle service call failed or timed out!")
            response.success = False
            response.message = "Toggle body pose estimation service call failed or timed out"
            return response

        # Verify the response
        toggle_service_response = future.result()
        if not toggle_service_response.success:
            self.get_logger().error("Toggle service call was not successful!")
            response.success = False
            response.message = "Toggle body pose estimation service call failed or timed out"
            return response

        # Enable or disable the body pose estimator
        with self._pose_estimator_enabled_lock:
            self._pose_estimator_enabled = request.data
        self._pose_history = []

        # Return success
        response.success = True
        response.message = "Success"
        return response

    def callback_body_landmarks(self, msg: String):
        self.get_logger().debug("Received body landmarks...", throttle_duration_sec=1.0)
        with self._pose_estimator_enabled_lock:
            pose_estimator_enabled = self._pose_estimator_enabled
        if pose_estimator_enabled:
            # config
            necessary_keys = [
                "nose",
                "neck",
                "right_shoulder",
                "left_shoulder"
            ]
        
            # Load data
            # msg_data = msg.data.replace("\"", "")
            # if msg_data == "None" or msg_data is None:
            #     return
            if len(msg.data) == 0:
                return

            # data = load_bad_json_data(msg_data)
            pose_camera_frame = json.loads(msg.data)

            if pose_camera_frame is None or len(pose_camera_frame.keys()) == 0:
                return
            
            # populate pose
            camera_pose = self.lookup_camera_pose()
            camera_pose = posestamped2se3(camera_pose)
            latest_pose = HumanPoseEstimate()
            latest_pose.set_body_estimate_camera_frame(pose_camera_frame, camera_pose)

            # get visible keypoints
            pose_keys = latest_pose.get_body_estimate_robot_frame().keys()

            # check if necessary keys visible
            can_see = True
            for key in necessary_keys:
                if key not in pose_keys:
                    self.get_logger().info("cannot see key joints!")
                    can_see = False
                    continue  # for
            
            if not can_see:
                return
            
            # publish pose and add to pose history
            # self.pub_valid_estimates.publish(String(data=msg_data))
            self.__push_to_pose_history(latest_pose)

            # TODO: Add a timestamp associated with the latest average pose.
            self.latest_average_pose = HumanPoseEstimate.average_pose_estimates(self.get_pose_history())
            # pose_str = self.latest_average_pose.get_body_estimate_string()
            pose_str = json.dumps(pose_camera_frame)
            self.pub_valid_estimates.publish(String(data=pose_str))

    # action callbacks
    # def callback_action_success(self):
    #     self.cleanup()
    #     self.result.status = ShowTablet.Result.STATUS_SUCCESS
    #     return ShowTablet.Result()

    def callback_action_goal(self, goal_request: ShowTablet.Goal) -> GoalResponse:
        """
        Accept a goal if this action does not already have an active goal, else reject.

        Parameters
        ----------
        goal_request: The goal request message.
        """
        self.get_logger().info(f"Received request {goal_request}")

        # Reject the goal if we don't have a valid pose estimate
        if self.latest_average_pose is None:
            self.get_logger().info(
                "Rejecting goal request since there is no valid pose estimate. "
                "Be sure to toggle detection on and ensure at least one message is "
                "sent on the /human_estimates/latest_body_pose topic before calling "
                "this action server."
            )
            return GoalResponse.REJECT

        # TODO: Reject the goal if the latest average pose is stale e.g., computed too long ago.

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
        return GoalResponse.ACCEPT

    def callback_action_cancel(self, goal_handle):
        self.get_logger().info("Received cancel request, accepting")
        self.cleanup()
        self.result.status = ShowTablet.Result.STATUS_CANCELED
        return CancelResponse.ACCEPT
        # return ShowTablet.Result(result=-1)

    # def callback_action_error(self):
    #     self.cleanup()
    #     self.result.status = ShowTablet.Result.STATUS_ERROR
    #     return ShowTablet.Result()

    # states
    def state_idle(self):
        if self.abort or self.goal_handle.is_cancel_requested:
            return ShowTabletState.ABORT

        if False:
            return ShowTabletState.IDLE
        
        return ShowTabletState.ESTIMATE_HUMAN_POSE
    
    def state_estimate_pose(self):
        self.get_logger().info("Estimating pose...")
        if self.abort or self.goal_handle.is_cancel_requested:
            return ShowTabletState.ABORT

        # Because the goal was accepted, we are guarenteed to have a non-None
        # `self.latest_average_pose` that is not stale (TODO).
        self.human.pose_estimate = self.latest_average_pose

        return ShowTabletState.PLAN_TABLET_POSE

    def state_plan_tablet_pose(self):
        self.get_logger().info("Planning tablet pose...")
        if self.abort or self.goal_handle.is_cancel_requested:
            return ShowTabletState.ABORT

        # Check if the service is available
        if not self.srv_plan_tablet_pose.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("Service not available!")
            return ShowTabletState.ERROR
        
        # if not self.human.pose_estimate.is_populated():
            # Human not seen
            # return ShowTabletState.LOOK_AT_HUMAN
        
        plan_request = self.build_tablet_pose_request()
        future = self.srv_plan_tablet_pose.call_async(plan_request)

        # wait for srv to finish
        self.get_logger().info("Waiting for srv to finish...")
        self.__wait_for_future(future)
        self.get_logger().info("Srv finished!")
        response = future.result()
        self.get_logger().info(f"Response received! {response} {type(response)}")

        if response.success:
            # get planner result
            joint_names = response.robot_ik_joint_names
            joint_positions = response.robot_ik_joint_positions

            # process output
            joint_dict = {n: p for n, p in zip(joint_names, joint_positions)}
            joint_dict = enforce_joint_limits(joint_dict)

            # robot cannot do small moves
            if abs(joint_dict["base"]) < 0.2:
                joint_dict.pop("base")

            # debug
            self.get_logger().info("JOINT CONFIG:")
            self.get_logger().info(json.dumps(joint_dict, indent=2))

            self._robot_joint_target = joint_dict

        # return ShowTabletState.NAVIGATE_BASE
        return ShowTabletState.MOVE_ARM_TO_TABLET_POSE

    def state_navigate_base(self):
        if self.abort or self.goal_handle.is_cancel_requested:
            return ShowTabletState.ABORT

        # TODO: Implement this state
        # Drive the robot to a human-centric location

        return ShowTabletState.MOVE_ARM_TO_TABLET_POSE
    
    def state_move_arm_to_tablet_pose(self):
        self.get_logger().info("Moving arm to tablet pose...")
        if self.abort or self.goal_handle.is_cancel_requested:
            return ShowTabletState.ABORT

        target = self._robot_joint_target
        self.__present_tablet(target)
        self.get_logger().info('Finished move_to_pose')

        return ShowTabletState.EXIT

    def state_end_interaction(self):
        self.get_logger().info("Ending interaction...")
        if self.abort or self.goal_handle.is_cancel_requested:
            return ShowTabletState.ABORT

        self.srv_toggle_navigation_mode.call(Trigger.Request())

        return ShowTabletState.EXIT
    
    def state_abort(self):
        # TODO: stop all motion
        return ShowTabletState.EXIT

    def state_exit(self):
        self.result.status = ShowTablet.Result.STATUS_SUCCESS
        return ShowTabletState.EXIT

    def run_state_machine(self, test:bool=False):
        self.get_logger().info("Running state machine...")
        state = self.state_idle()

        while rclpy.ok():
            self.get_logger().info(f"Current state: {state.name}")
            # TODO: Handle ERROR and ABORT better. Currently ERROR loops
            # back to IDLE, and ABORT sets the result to SUCCESS.
            # check abort
            if state == ShowTabletState.ABORT:
                state = self.state_abort()
                break
            # main state machine
            elif state == ShowTabletState.IDLE:
                state = self.state_idle()
            elif state == ShowTabletState.ESTIMATE_HUMAN_POSE:
                state = self.state_estimate_pose()
            elif state == ShowTabletState.PLAN_TABLET_POSE:
                state = self.state_plan_tablet_pose()
            # elif state == ShowTabletState.NAVIGATE_BASE:
            #     state = self.state_navigate_base()
            elif state == ShowTabletState.MOVE_ARM_TO_TABLET_POSE:
                state = self.state_move_arm_to_tablet_pose()
            # elif state == ShowTabletState.TRACK_FACE:
            #     state = self.state_track_face()
            elif state == ShowTabletState.END_INTERACTION:
                state = self.state_end_interaction()
            elif state == ShowTabletState.EXIT:
                state = self.state_exit()
                break
            else:
                state = ShowTabletState.IDLE

            self.feedback.current_state = state.value
            self.goal_handle.publish_feedback(self.feedback)
            if test:
                time.sleep(0.25)
        
        return self.result

    def execute_callback(self, goal_handle):
        # TODO: Add a timeout!
        self.get_logger().info('Executing Show Tablet...')
        self.goal_handle = goal_handle

        # load body pose
        # pose_estimate = load_bad_json_data(goal_handle.request.human_joint_dict)
        # self.human.pose_estimate.set_body_estimate(pose_estimate)
        # self.set_human_camera_pose(goal_handle.request.camera_pose)

        # self.get_logger().info(str(self.human.pose_estimate.get_body_world()))
        # self.get_logger().info(str(self.human.pose_estimate.get_camera_pose()))
        
        final_result = self.run_state_machine(test=True)

        # get result
        if final_result.status == ShowTablet.Result.STATUS_SUCCESS:
            goal_handle.succeed()
        elif final_result.status == ShowTablet.Result.STATUS_ERROR:
            goal_handle.abort()
        elif final_result.status == ShowTablet.Result.STATUS_CANCELED:
            goal_handle.canceled()
        else:
            raise ValueError
        
        self.get_logger().info(f"Final result: {final_result}")
        self.active_goal_request = None
        return final_result


def main(args=None):
    rclpy.init(args=args)

    # Initialize the action server
    show_tablet_action_server = ShowTabletActionServer()
    show_tablet_action_server.get_logger().info("ShowTabletActionServer initialized")

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()
    rclpy.spin(show_tablet_action_server, executor=executor)

    # Cleanup
    show_tablet_action_server.destroy()


if __name__ == '__main__':
    main()
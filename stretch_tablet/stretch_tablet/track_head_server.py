import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse, ActionClient
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.task import Future
from ament_index_python import get_package_share_directory

import tf2_ros

from std_msgs.msg import String, Bool
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from stretch_tablet_interfaces.action import TrackHead

from stretch_tablet.human import HumanPoseEstimate
from stretch_tablet.utils import load_bad_json_data
from stretch_tablet.control import TabletController
import json
import os

from typing import Optional, Tuple
from enum import Enum

# from stretch_tablet.stretch_ik_control import StretchIKControl

def in_range(value, range):
    return True if value >= range[0] and value <= range[1] else False

class HeadTrackerState(Enum):
    IDLE = 0
    TRACKING = 1
    CANNOT_SEE = 2
    DONE = 3
    ERROR = 99

class HeadTrackerServer(Node):
    def __init__(self):
        super().__init__('head_tracker_action_server')
        self._action_server = ActionServer(
            self,
            TrackHead,
            'track_head',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            cancel_callback=self.callback_action_cancel)

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
        self._pose_estimate = HumanPoseEstimate()
        self.controller = TabletController()

        # simpler web teleop
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # # web teleop stuff
        # tf_timeout_secs = 0.5

        # # Initialize TF2
        # self.tf_timeout = Duration(seconds=tf_timeout_secs)
        # self.tf_buffer = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # self.static_transform_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        # self.lift_offset: Optional[Tuple[float, float]] = None
        # self.wrist_offset: Optional[Tuple[float, float]] = None

        # # Create the inverse jacobian controller to execute motions
        # urdf_abs_path = os.path.join(
        #     get_package_share_directory("stretch_web_teleop"),
        #     "urdf/stretch_base_rotation_ik.urdf",
        # )

        # self.ik_control = StretchIKControl(
        #     self,
        #     tf_buffer=self.tf_buffer,
        #     urdf_path=urdf_abs_path,
        #     static_transform_broadcaster=self.static_transform_broadcaster,
        # )

        # config
        self.debug = False

    # callbacks
    def callback_action_cancel(self, goal_handle: ServerGoalHandle):
        pass
    
    def callback_face_landmarks(self, msg):        
        msg_data = msg.data.replace("\"", "")
        if msg_data == "None" or msg_data is None:
            return

        data = load_bad_json_data(msg_data)

        if data is None:
            return

        # update human estimate
        self._pose_estimate.set_face_estimate(data)

        # compute yaw action and  send command
        # yaw_action = self.controller.get_tablet_yaw_action(self.human)
        # move_msg = String()
        # move_msg.data = str(json.dumps({'joint_wrist_yaw': yaw_action}))
        # self.pub_tablet_move_by.publish(move_msg)

        # # debug
        # self.print(str(yaw_action))
        # self.print(str(self.controller.get_head_vertical_vector(self.human)))
        # self.print(str(self.controller.get_head_direction(self.human)))

    # state machine
    def state_idle(self):
        return HeadTrackerState.TRACKING
    
    def state_track_head(self):
        return HeadTrackerState.DONE
    
    def state_cannot_see_head(self):
        return HeadTrackerState.CANNOT_SEE

    def state_done(self):
        self.__command_move_wrist(0.)
        return HeadTrackerState.DONE

    def run_state_machine(self, goal_handle: ServerGoalHandle):
        state = HeadTrackerState.IDLE
        feedback = TrackHead.Feedback()
        result = TrackHead.Result()
        rate = self.create_rate(1.)

        while rclpy.ok():
            feedback.status = state.value
            goal_handle.publish_feedback(feedback)
            if state == HeadTrackerState.IDLE:
                state = self.state_idle()
            elif state == HeadTrackerState.TRACKING:
                state = self.state_track_head()
            elif state == HeadTrackerState.CANNOT_SEE:
                state = self.state_cannot_see_head()
            elif state == HeadTrackerState.DONE:
                state = self.state_done()
                goal_handle.succeed()
                break
            else:
                state = HeadTrackerState.IDLE

            rate.sleep()

        return result

    def execute_callback(self, goal_handle: ServerGoalHandle):
        return self.run_state_machine(goal_handle)

    # helpers
    def __command_move_wrist(self, yaw_position) -> Future:
        # Create the goal
        wrist_goal = FollowJointTrajectory.Goal()

        # joint info
        wrist_goal.trajectory.joint_names = ["joint_wrist_yaw"]

        # 
        wrist_goal.trajectory.points = [JointTrajectoryPoint()]
        wrist_goal.trajectory.points[0].positions = [yaw_position]
        wrist_goal.trajectory.points[0].time_from_start = Duration(
            seconds=1.,
        ).to_msg()

        return self.arm_client.send_goal_async(wrist_goal)

    def print(self, msg):
        if self.debug:
            self.get_logger().info(str(msg))

    # main
    def main(self):
        executor = MultiThreadedExecutor()
        rclpy.spin(self, executor=executor)

def main():
    rclpy.init()
    HeadTrackerServer().main()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

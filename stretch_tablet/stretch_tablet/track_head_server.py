import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse, ActionClient
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.task import Future


from std_msgs.msg import String, Bool
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState

from stretch_tablet_interfaces.action import TrackHead

from stretch_tablet.human import HumanPoseEstimate
from stretch_tablet.utils import load_bad_json_data
from stretch_tablet.control import TabletController

from enum import Enum

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
        # TODO: publish toggles for the face tracker

        # sub
        self.sub_face_landmarks = self.create_subscription(
            String,
            "/faces_gripper/landmarks_3d",
            callback=self.callback_face_landmarks,
            qos_profile=1
        )

        self.sub_joint_state = self.create_subscription(
            JointState,
            "/joint_states",
            callback=self.callback_joint_state,
            qos_profile=1
        )

        # state
        self._pose_estimate = HumanPoseEstimate()
        self._joint_state = JointState()
        self._exit = False

        self.controller = TabletController()

        # simpler web teleop
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.debug = False

    # callbacks
    def callback_action_cancel(self, goal_handle: ServerGoalHandle):
        self._exit = True
        return CancelResponse.ACCEPT
    
    def callback_face_landmarks(self, msg):        
        msg_data = msg.data.replace("\"", "")
        if msg_data == "None" or msg_data is None:
            return

        data = load_bad_json_data(msg_data)

        if data is None:
            return

        # update human estimate
        self._pose_estimate.set_face_estimate(data)

    def callback_joint_state(self, msg: JointState):
        self._joint_state = msg

    # state machine
    def state_idle(self):
        return HeadTrackerState.TRACKING
    
    def state_track_head(self):
        if self._pose_estimate.face_estimate is None:
            return HeadTrackerState.IDLE
        
        # TODO: check if cannot_see_head

        try:
            wrist_yaw_angle = self._joint_state.position[self._joint_state.name.index("joint_wrist_yaw")]
        except ValueError as e:
            self.get_logger().error(str(e))
            wrist_yaw_angle = 0.

        wrist_delta = self.controller.get_tablet_yaw_from_head_pose(self._pose_estimate)
        self.__command_move_wrist(wrist_yaw_angle + wrist_delta)
        return HeadTrackerState.TRACKING
    
    def state_cannot_see_head(self):
        # TODO: populate
        return HeadTrackerState.CANNOT_SEE

    def state_done(self):
        self.__command_move_wrist(0., move_time_s=2.)
        return HeadTrackerState.DONE

    def run_state_machine(self, goal_handle: ServerGoalHandle):
        self.state = HeadTrackerState.IDLE
        feedback = TrackHead.Feedback()
        result = TrackHead.Result()
        rate = self.create_rate(1.)

        while rclpy.ok():
            self.get_logger().info("Current State: " + str(self.state))
            if self._exit:
                self.state = HeadTrackerState.DONE

            feedback.status = self.state.value
            goal_handle.publish_feedback(feedback)
            if self.state == HeadTrackerState.IDLE:
                new_state = self.state_idle()
            elif self.state == HeadTrackerState.TRACKING:
                new_state = self.state_track_head()
            elif self.state == HeadTrackerState.CANNOT_SEE:
                new_state = self.state_cannot_see_head()
            elif self.state == HeadTrackerState.DONE:
                new_state = self.state_done()
                goal_handle.succeed()
                break
            else:
                self.state = HeadTrackerState.IDLE

            self.state = new_state
            rate.sleep()

        return result

    def execute_callback(self, goal_handle: ServerGoalHandle):
        self.get_logger().info('Executing Track Head...')
        return self.run_state_machine(goal_handle)

    # helpers
    def __command_move_wrist(self, yaw_position, move_time_s: float=1.) -> Future:
        # Create the goal
        wrist_goal = FollowJointTrajectory.Goal()

        # joint info
        wrist_goal.trajectory.joint_names = ["joint_wrist_yaw"]

        # populate positions and times
        wrist_goal.trajectory.points = [JointTrajectoryPoint()]
        wrist_goal.trajectory.points[0].positions = [yaw_position]
        wrist_goal.trajectory.points[0].time_from_start = Duration(
            seconds=move_time_s,
        ).to_msg()

        return self.arm_client.send_goal_async(wrist_goal)

    def print(self, msg):
        """
        helper to reduce keystrokes required to debug print to terminal >:(
        """
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

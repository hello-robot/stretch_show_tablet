import threading
from enum import Enum

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.timer import Rate
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectoryPoint

from stretch_show_tablet_interfaces.action import ShowTablet


def run_action(
    action_handle: ActionClient, request, rate: Rate, feedback_callback=None
):
    """
    Runs an action and waits for the result.

    Args:
        action_handle (ActionClient): action client handle
        request (Any): action request
        rate (Rate): rate object
        feedback_callback (Callable): callback for feedback

    Returns:
        Any: action result
    """

    future = action_handle.send_goal_async(request, feedback_callback=feedback_callback)

    # wait for server to accept goal
    while rclpy.ok():
        if future.done():
            break

        rate.sleep()

    # wait for result
    goal_handle = future.result()
    result_future = goal_handle.get_result_async()
    while rclpy.ok() and not result_future.done():
        rate.sleep()

    result = result_future.result().result
    return result


class DemoState(Enum):
    """
    States for the demo state machine.
    """

    IDLE = 0
    TOGGLE_DETECTION = 1
    SHOW_TABLET = 2
    RETRACT_ARM = 3
    JOG_HEAD = 4
    EXIT = 99


class DemoShowTablet(Node):
    def __init__(self):
        super().__init__("demo_show_tablet")

        # services
        self.srv_toggle_detection = self.create_client(
            SetBool,
            "/toggle_body_pose_estimator",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # actions
        self.act_show_tablet = ActionClient(
            self,
            ShowTablet,
            "show_tablet",
        )

        self.act_move_arm = ActionClient(
            self,
            FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            callback=self.callback_joint_state,
            qos_profile=1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # wait for servers
        _wait_time = 10.0
        if not self.act_show_tablet.wait_for_server(_wait_time):
            self.get_logger().error(
                "DemoShowTablet::init: did not find action server, exiting..."
            )
            rclpy.shutdown()

        # config
        self._wait_rate_hz = 10.0

        # feedback
        self._feedback_show_tablet = ShowTablet.Feedback()

        # state
        self._toggle_detection = False
        self._joint_state = JointState()

    # callbacks
    def callback_show_tablet_feedback(self, feedback: ShowTablet.Feedback):
        self._feedback_show_tablet = feedback
        self.get_logger().info("Show Tablet Feedback: " + str(feedback))

    def callback_joint_state(self, msg: JointState):
        self._joint_state = msg

    # helpers
    def create_retract_arm_goal(
        self,
        move_time_sec: float = 4.0,
    ) -> FollowJointTrajectory.Goal:
        """
        Generates a FollowJointTrajectory goal to retract the arm to 0.05m.

        Args:
            move_time_sec (float): time to complete the move

        Returns:
            FollowJointTrajectory.Goal: goal
        """

        joint_positions = [0.05]
        trajectory = FollowJointTrajectory.Goal()
        trajectory.trajectory.joint_names = ["wrist_extension"]
        trajectory.trajectory.points = [JointTrajectoryPoint()]
        trajectory.trajectory.points[0].positions = [p for p in joint_positions]
        trajectory.trajectory.points[0].time_from_start = Duration(
            seconds=move_time_sec
        ).to_msg()
        return trajectory

    def create_jog_head_goal(
        self,
        delta_tilt: float,
        delta_pan: float,
        move_time_sec: float = 0.5,
    ) -> FollowJointTrajectory.Goal:
        """
        Generates a FollowJointTrajectory goal to jog the head.

        Args:
            delta_tilt (float): change in tilt (radians)
            delta_pan (float): change in pan (radians)
            move_time_sec (float): time to complete the move

        Returns:
            FollowJointTrajectory.Goal: goal
        """

        # get current head position
        current_tilt = self._joint_state.position[
            self._joint_state.name.index("joint_head_tilt")
        ]
        current_pan = self._joint_state.position[
            self._joint_state.name.index("joint_head_pan")
        ]

        joint_positions = [current_tilt + delta_tilt, current_pan + delta_pan]

        trajectory = FollowJointTrajectory.Goal()
        trajectory.trajectory.joint_names = ["joint_head_tilt", "joint_head_pan"]
        trajectory.trajectory.points = [JointTrajectoryPoint()]
        trajectory.trajectory.points[0].positions = [p for p in joint_positions]
        trajectory.trajectory.points[0].time_from_start = Duration(
            seconds=move_time_sec
        ).to_msg()
        return trajectory

    # states
    def state_idle(self) -> DemoState:
        """
        Prints the main menu and waits for user input.

        Returns:
            DemoState: next state
        """

        print(" ")
        print("=" * 5 + " Main Menu " + 5 * "=")
        print("(T) Toggle Pose Estimator    (S) Show Tablet")
        print("(R) Retract Arm              (J) Jog Head")
        print("(Q) Quit")
        ui = input("Selection:").lower()

        if ui == "t":
            return DemoState.TOGGLE_DETECTION
        elif ui == "s":
            return DemoState.SHOW_TABLET
        elif ui == "r":
            return DemoState.RETRACT_ARM
        elif ui == "j":
            return DemoState.JOG_HEAD
        elif ui == "q":
            return DemoState.EXIT
        else:
            return DemoState.IDLE

    def state_toggle_detection(self) -> DemoState:
        """
        Sends a blocking request to toggle the pose estimator.

        Returns:
            DemoState: next state
        """

        self.get_logger().info("DemoShowTablet: Toggling Pose Estimator...")

        # send request
        request = SetBool.Request()
        request.data = not self._toggle_detection

        self.srv_toggle_detection.call(request)

        self.get_logger().info(
            "DemoShowTablet: Pose Estimator set to " + str(not self._toggle_detection)
        )

        self._toggle_detection = not self._toggle_detection

        return DemoState.IDLE

    def state_show_tablet(self) -> DemoState:
        """
        Sends a blocking request to show the tablet.

        Returns:
            DemoState: next state
        """

        if not self._toggle_detection:
            self.get_logger().error("DemoShowTablet: Pose Estimator is not running")
            return DemoState.IDLE

        self.get_logger().info("DemoShowTablet: Showing Tablet...")

        # send request
        request = ShowTablet.Goal()
        request.number_of_pose_estimates = 10

        result = run_action(  # noqa: F841
            self.act_show_tablet,
            request,
            self.rate,
            feedback_callback=self.callback_show_tablet_feedback,
        )

        return DemoState.IDLE

    def state_retract_arm(self) -> DemoState:
        """
        Sends a blocking request to retract the arm.

        Returns:
            DemoState: next state
        """

        self.get_logger().info("DemoShowTablet: Retracting Arm...")
        # send request
        request = self.create_retract_arm_goal()

        result = run_action(  # noqa: F841
            self.act_move_arm,
            request,
            self.rate,
        )

        return DemoState.IDLE

    def state_jog_head(self, print_tip: bool = True) -> DemoState:
        """
        Prints the jog head menu and waits for user input.
        Allows the user to jog the robot's head to point at a person.

        Returns:
            DemoState: next state
        """

        print(" ")
        print("=" * 5 + " Jog Head Menu " + 5 * "=")
        print("(I) Up      (K) Down")
        print("(J) Left    (L) Right")
        print("(Q) Quit")

        if print_tip:
            print(" ")
            print("(!) Tip (!)")
            print("Point the head at a person to improve pose estimation.")
            print(" ")

        ui = input("Selection:").lower()

        if ui == "q":
            return DemoState.IDLE

        delta_tilt = 0.0
        delta_pan = 0.0

        if ui == "i":
            delta_tilt = 0.1
        elif ui == "k":
            delta_tilt = -0.1
        elif ui == "j":
            delta_pan = 0.1
        elif ui == "l":
            delta_pan = -0.1

        # send request
        if abs(delta_tilt) > 0.0 or abs(delta_pan) > 0.0:
            request = self.create_jog_head_goal(
                delta_tilt=delta_tilt, delta_pan=delta_pan
            )

        result = run_action(  # noqa: F841
            self.act_move_arm,
            request,
            self.rate,
        )

        return DemoState.JOG_HEAD

    def state_exit(self) -> DemoState:
        """
        Exit and cleanup.

        Returns:
            DemoState: next state
        """
        self.get_logger().info("DemoShowTablet: Exiting!")
        return DemoState.EXIT

    def run(self):
        """
        Main state machine loop.
        """

        state = DemoState.IDLE

        self.rate = self.create_rate(self._wait_rate_hz)

        while rclpy.ok():
            self.get_logger().info("Current State: " + str(state))
            if state == DemoState.IDLE:
                state = self.state_idle()
            elif state == DemoState.TOGGLE_DETECTION:
                state = self.state_toggle_detection()
            elif state == DemoState.SHOW_TABLET:
                state = self.state_show_tablet()
            elif state == DemoState.RETRACT_ARM:
                state = self.state_retract_arm()
            elif state == DemoState.JOG_HEAD:
                state = self.state_jog_head()
            elif state == DemoState.EXIT:
                state = self.state_exit()
                break
            else:
                state = DemoState.IDLE

            self.rate.sleep()

        self.get_logger().info("DemoShowTablet: Done.")


def main():
    rclpy.init()
    node = DemoShowTablet()
    executor = MultiThreadedExecutor(num_threads=4)

    # Spin in the background since node has blocking behavior
    # the main thread
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        kwargs={"executor": executor},
        daemon=True,
    )
    spin_thread.start()

    # Run demo
    try:
        node.run()
    except KeyboardInterrupt:
        pass

    # Terminate this node
    node.destroy_node()
    rclpy.shutdown()
    # Join the spin thread (so it is spinning in the main thread)
    spin_thread.join()


if __name__ == "__main__":
    main()

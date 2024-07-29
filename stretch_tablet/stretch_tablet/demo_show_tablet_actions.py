import threading
from enum import Enum

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.timer import Rate

from stretch_tablet_interfaces.action import EstimateHumanPose, ShowTablet, TrackHead


def run_action(
    action_handle: ActionClient, request, rate: Rate, feedback_callback=None
):
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
    IDLE = 0
    # ESTIMATE_POSE = 1
    SHOW_TABLET = 2
    TRACK_HEAD = 3
    EXIT = 99


class DemoShowTablet(Node):
    def __init__(self):
        super().__init__("demo_show_tablet")
        # actions
        self.act_estimate_pose = ActionClient(
            self,
            EstimateHumanPose,
            "estimate_human_pose",
        )
        self.act_show_tablet = ActionClient(
            self,
            ShowTablet,
            "show_tablet",
        )
        self.act_track_head = ActionClient(
            self,
            TrackHead,
            "track_head",
        )

        # wait for servers
        _wait_time = 10.0
        for act in [self.act_show_tablet, self.act_track_head]:
            if not act.wait_for_server(_wait_time):
                self.get_logger().error(
                    "DemoShowTablet::init: did not find action servers, exiting..."
                )
                rclpy.shutdown()

        # config
        self._wait_rate_hz = 10.0

        # state
        self._body_pose_estimate = None
        self._camera_pose = None

        # feedback
        self._feedback_estimate_pose = EstimateHumanPose.Feedback()
        self._feedback_show_tablet = ShowTablet.Feedback()
        self._feedback_track_head = TrackHead.Feedback()

    # callbacks
    def callback_show_tablet_feedback(self, feedback: ShowTablet.Feedback):
        self._feedback_show_tablet = feedback

    def callback_track_head_feedback(self, feedback: TrackHead.Feedback):
        self._feedback_track_head = feedback

    # states
    def state_idle(self) -> DemoState:
        print(" ")
        print("=" * 5 + " Main Menu " + 5 * "=")
        # print("(E) Estimate Pose    (Q) Quit")
        print("(S) Show Tablet    (Q) Quit")
        ui = input("Selection:").lower()

        if ui == "s":
            return DemoState.SHOW_TABLET
        elif ui == "q":
            return DemoState.EXIT
        else:
            return DemoState.IDLE

    def state_show_tablet(self) -> DemoState:
        # if self._body_pose_estimate is None:
        #     self.get_logger().warn("DemoShowTablet::state_show_tablet: body estimate is None!")
        #     return DemoState.ESTIMATE_POSE

        # send request
        request = ShowTablet.Goal()
        # request.human_joint_dict = json.dumps(self._body_pose_estimate)
        # request.camera_pose = self._camera_pose

        result = run_action(  # noqa: F841
            self.act_show_tablet,
            request,
            self.rate,
            feedback_callback=self.callback_show_tablet_feedback,
        )

        return DemoState.TRACK_HEAD

    def state_track_head(self) -> DemoState:
        # send request
        request = TrackHead.Goal()

        result = run_action(  # noqa: F841
            self.act_track_head,
            request,
            self.rate,
            feedback_callback=self.callback_track_head_feedback,
        )

        return DemoState.EXIT

    def state_exit(self) -> DemoState:
        self.get_logger().info("DemoShowTablet: Exiting!")
        return DemoState.EXIT

    def run(self):
        state = DemoState.IDLE

        self.rate = self.create_rate(self._wait_rate_hz)

        while rclpy.ok():
            self.get_logger().info("Current State: " + str(state))
            if state == DemoState.IDLE:
                state = self.state_idle()
            # elif state == DemoState.ESTIMATE_POSE:
            #     state = self.state_estimate_human_pose()
            elif state == DemoState.SHOW_TABLET:
                state = self.state_show_tablet()
            elif state == DemoState.TRACK_HEAD:
                state = self.state_track_head()
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

    # Spin in the background since detecting faces will block
    # the main thread
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        kwargs={"executor": executor},
        daemon=True,
    )
    spin_thread.start()

    # Run face detection
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

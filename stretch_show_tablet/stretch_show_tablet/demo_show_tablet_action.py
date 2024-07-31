import threading
from enum import Enum

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.timer import Rate

from stretch_show_tablet_interfaces.action import ShowTablet


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
    EXIT = 99


class DemoShowTablet(Node):
    def __init__(self):
        super().__init__("demo_show_tablet")

        # actions
        self.act_show_tablet = ActionClient(
            self,
            ShowTablet,
            "show_tablet",
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

    # callbacks
    def callback_show_tablet_feedback(self, feedback: ShowTablet.Feedback):
        self._feedback_show_tablet = feedback

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
            elif state == DemoState.SHOW_TABLET:
                state = self.state_show_tablet()
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

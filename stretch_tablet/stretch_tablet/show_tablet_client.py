import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from stretch_tablet_interfaces.action import ShowTablet
from stretch_tablet.show_tablet_server import ShowTabletState

class ShowTabletActionClient(Node):
    def __init__(self):
        super().__init__('show_tablet_action_client')
        self._action_client = ActionClient(self, ShowTablet, 'show_tablet')

    def send_goal(self):
        goal_msg = ShowTablet.Goal()
        goal_msg.request = 0

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.result))
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        try:
            state = ShowTabletState(feedback.current_state)
        except ValueError as e:
            print(e)
            state = -1
        
        self.get_logger().info('Current State: {0}'.format(state))

def main(args=None):
    rclpy.init(args=args)

    action_client = ShowTabletActionClient()
    action_client.send_goal()

    rclpy.spin(action_client)


if __name__ == '__main__':
    main()

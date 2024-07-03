import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from stretch_tablet_action_interface.action import ShowTablet


class ShowTabletActionServer(Node):

    def __init__(self):
        super().__init__('show_tablet_action_server')
        self._action_server = ActionServer(
            self,
            ShowTablet,
            'show_tablet',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing Show Tablet Plan...')
        result = ShowTablet.Result()
        return result


def main(args=None):
    rclpy.init(args=args)

    show_tablet_action_server = ShowTabletActionServer()

    rclpy.spin(show_tablet_action_server)


if __name__ == '__main__':
    main()
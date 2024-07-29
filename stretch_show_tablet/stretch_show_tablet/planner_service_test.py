import json

import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from stretch_show_tablet.human import generate_test_human
from stretch_show_tablet_interfaces.srv import PlanTabletPose


class TestPlannerClient(Node):
    def __init__(self):
        super().__init__("test_planner_client")
        self.cli = self.create_client(PlanTabletPose, "plan_tablet_pose")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        self.req = PlanTabletPose.Request()

    def send_request(self):
        # generate test human
        human = generate_test_human(
            "/home/hello-robot/ament_ws/src/stretch_show_tablet/data/matt/"
        )

        # extract request info
        body_string = json.dumps(human.pose_estimate.body_estimate)
        camera_pose = human.pose_estimate.get_camera_pose()
        camera_position = camera_pose.translation()
        camera_orientation = R.from_matrix(camera_pose.rotationMatrix()).as_quat()

        # construct request
        self.req.human_joint_dict = body_string
        self.req.camera_position = [v for v in camera_position]
        self.req.camera_orientation = [v for v in camera_orientation]
        self.future = self.cli.call_async(self.req)


def main(args=None):
    rclpy.init(args=args)

    client = TestPlannerClient()
    client.send_request()

    while rclpy.ok():
        rclpy.spin_once(client)
        if client.future.done():
            try:
                response = client.future.result()
            except Exception as e:
                client.get_logger().info("Service call failed %r" % (e,))
            else:
                client.get_logger().info("Result of tablet plan:")
                client.get_logger().info(str(response.tablet_position))
                client.get_logger().info(str(response.tablet_orientation))
            break

    client.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

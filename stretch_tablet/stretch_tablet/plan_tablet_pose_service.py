from stretch_tablet_interfaces.srv import PlanTabletPose

import rclpy
from rclpy.node import Node

import sophuspy as sp
from scipy.spatial.transform import Rotation as R

from stretch_tablet.planner import TabletPlanner
from stretch_tablet.human import Human

import json

class PlanTabletPoseService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(PlanTabletPose, 'plan_tablet_pose', self.plan_tablet_callback)
        self.planner = TabletPlanner()

    def plan_tablet_callback(self, request, response):
        # generate human
        body_dict = json.loads(request.human_joint_dict)
        camera_position = request.camera_position
        camera_orientation = R.from_quat(request.camera_orientation).as_matrix()
        camera_transform = sp.SE3(camera_orientation, camera_position)

        human = Human()
        human.pose_estimate.set_body_estimate(body_dict)
        human.pose_estimate.set_camera_pose(camera_transform)

        # run planner
        tablet_pose_world = self.planner.in_front_of_eyes(human)
        tablet_position = tablet_pose_world.translation()
        tablet_orientation = R.from_matrix(tablet_pose_world.rotationMatrix()).as_quat()
        # self.get_logger().info(str(tablet_position))
        # self.get_logger().info(str(tablet_orientation))

        # get robot base
        robot_position = request.robot_position_world
        robot_orientation = R.from_quat(request.robot_orientation_world).as_matrix()
        robot_pose = sp.SE3(robot_orientation, robot_position)

        # solve ik
        ik_solution, _ = self.planner.ik(
            world_target=tablet_pose_world,
            world_base_link=robot_pose
            )

        # save response
        response.tablet_position_robot_frame = [v for v in tablet_position]
        response.tablet_orientation_robot_frame = [v for v in tablet_orientation]
        response.robot_ik_solution_dict = json.dumps(ik_solution)

        return response

def main(args=None):
    rclpy.init(args=args)
    service = PlanTabletPoseService()
    rclpy.spin(service)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
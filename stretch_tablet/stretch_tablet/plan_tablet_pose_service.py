import rclpy.time
from stretch_tablet_interfaces.srv import PlanTabletPose

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion

import sophuspy as sp
from scipy.spatial.transform import Rotation as R

from stretch_tablet.planner import TabletPlanner
from stretch_tablet.human import Human

import json
import time

from stretch_tablet.utils_ros import point2tuple, quat2tuple, generate_pose_stamped

# TODO: I got the below error somewhere in the service callback which caused the node to crash.
# Errors like this should get caught and the service should cleanly return a failure response.
# [plan_tablet_pose_service-29] /home/hello-robot/ament_ws/install/stretch_tablet/lib/python3.10/site-packages/stretch_tablet/planner.py:87: RuntimeWarning: invalid value encountered in divide
# [plan_tablet_pose_service-29]   x = x / np.linalg.norm(x)
# [plan_tablet_pose_service-29] /home/hello-robot/ament_ws/install/stretch_tablet/lib/python3.10/site-packages/stretch_tablet/planner.py:88: RuntimeWarning: invalid value encountered in divide
# [plan_tablet_pose_service-29]   y = y / np.linalg.norm(y)
# [plan_tablet_pose_service-29] Sophus ensure failed in function 'Sophus::SO3<Scalar_, Options>::SO3(const Transformation&) [with Scalar_ = double; int Options = 0; Sophus::SO3<Scalar_, Options>::Transformation = Eigen::Matrix<double, 3, 3>]', file 'sophuspy/include/original/so3.hpp', line 424.
# [plan_tablet_pose_service-29] R is not orthogonal:
# [plan_tablet_pose_service-29]  -nan -nan -nan
# [plan_tablet_pose_service-29] -nan -nan -nan
# [plan_tablet_pose_service-29] -nan -nan -nan

class PlanTabletPoseService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(PlanTabletPose, 'plan_tablet_pose', self.plan_tablet_callback)
        self.planner = TabletPlanner()

    def now(self):
        return self.get_clock().now().to_msg()

    def plan_tablet_callback(self, request, response):
        plan_start_time = time.time()
        # generate human
        body_dict = json.loads(request.human_joint_dict)
        camera_position = point2tuple(request.camera_pose.pose.position)
        camera_orientation = R.from_quat(quat2tuple(request.camera_pose.pose.orientation)).as_matrix()
        camera_transform = sp.SE3(camera_orientation, camera_position)

        human = Human()
        human.pose_estimate.set_body_estimate(body_dict)
        human.pose_estimate.set_camera_pose(camera_transform)

        # run planner
        tablet_pose_world = self.planner.in_front_of_eyes(human)
        if tablet_pose_world is None:
            response.success = False
            return response
        tablet_position = tablet_pose_world.translation()
        tablet_orientation = R.from_matrix(tablet_pose_world.rotationMatrix()).as_quat().tolist()
        # self.get_logger().info(str(tablet_position))
        # self.get_logger().info(str(tablet_orientation))

        # optimize base location
        base_location_world = self.planner.get_base_location(
            handle_cost_function=self.planner.cost_midpoint_displacement,
            tablet_pose_world=tablet_pose_world
        )

        # get robot base
        robot_position = point2tuple(request.robot_pose.pose.position)
        robot_orientation = R.from_quat(quat2tuple(request.robot_pose.pose.orientation)).as_matrix()
        robot_pose = sp.SE3(robot_orientation, robot_position)

        # solve ik
        ik_solution, _ = self.planner.ik(
            world_target=tablet_pose_world,
            world_base_link=robot_pose
            )

        # save response
        response.tablet_pose = generate_pose_stamped(tablet_position, tablet_orientation, self.now())
        response.robot_ik_joint_names = [k for k in ik_solution.keys()]
        response.robot_ik_joint_positions = [v for v in ik_solution.values()]
        # response.robot_base_pose_xy = [v for v in base_location_world]
        response.plan_time_s = time.time() - plan_start_time
        response.success = True

        return response

def main(args=None):
    rclpy.init(args=args)
    node = PlanTabletPoseService()
    node.get_logger().info('Plan Tablet Pose Service has been started.')
    rclpy.spin(node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
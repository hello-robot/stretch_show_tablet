import json
import time

import numpy as np
import rclpy
import rclpy.time
import sophuspy as sp
from geometry_msgs.msg import Transform, TransformStamped, Vector3
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from tf2_ros import StaticTransformBroadcaster

from stretch_show_tablet.human import Human
from stretch_show_tablet.planner import TabletPlanner
from stretch_show_tablet.planner_helpers import JOINT_LIMITS
from stretch_show_tablet.utils_ros import generate_pose_stamped
from stretch_show_tablet_interfaces.srv import PlanTabletPose


class PlanTabletPoseService(Node):
    def __init__(self):
        super().__init__("minimal_service")
        self.srv = self.create_service(
            PlanTabletPose, "plan_tablet_pose", self.plan_tablet_callback
        )
        self.planner = TabletPlanner()
        self.static_transform_broadcaster = StaticTransformBroadcaster(self)
        self.max_ik_error = 0.01

    def now(self):
        return self.get_clock().now().to_msg()

    def broadcast_static_tf(self, human_root: sp.SE3, tablet_pose: sp.SE3):
        for pose, child_frame in zip(
            [human_root, tablet_pose], ["human_root", "tablet_pose"]
        ):
            pose_msg = generate_pose_stamped(
                pose.translation(),
                R.from_matrix(pose.rotationMatrix()).as_quat().tolist(),
                self.now(),
            )
            pose_msg.header.frame_id = "base_link"

            self.static_transform_broadcaster.sendTransform(
                TransformStamped(
                    header=pose_msg.header,
                    child_frame_id=child_frame,
                    transform=Transform(
                        translation=Vector3(
                            x=pose_msg.pose.position.x,
                            y=pose_msg.pose.position.y,
                            z=pose_msg.pose.position.z,
                        ),
                        rotation=pose_msg.pose.orientation,
                    ),
                )
            )

    def plan_tablet_callback(self, request, response):
        plan_start_time = time.time()
        # generate human
        body_dict = json.loads(request.human_joint_dict_robot_frame)
        human = Human()
        human.pose_estimate.set_body_estimate_robot_frame(body_dict)

        # run planner
        try:
            (
                tablet_pose_robot_frame,
                human_head_root,
            ) = self.planner.in_front_of_eyes(human)
        except Exception as e:
            print("PlanTabletPoseService::plan_tablet_callback: " + str(e))
            response.success = False
            return response

        if tablet_pose_robot_frame is None:
            response.success = False
            return response

        tablet_position = np.array(tablet_pose_robot_frame.translation())
        tablet_orientation = (
            R.from_matrix(tablet_pose_robot_frame.rotationMatrix()).as_quat().tolist()
        )

        # broadcast
        self.broadcast_static_tf(human_head_root, tablet_pose_robot_frame)

        # TODO: implement this after base localization works well
        # NOTE: the IK + sampling methods in planner.py need some TLC to get working again
        # optimize base location
        # base_location_world = self.planner.get_base_location(
        #     handle_cost_function=self.planner.cost_midpoint_displacement,
        #     tablet_pose_world=tablet_pose_world
        # )
        # response.robot_base_pose_xy = [v for v in base_location_world]

        # solve IK
        ik_solution, _ = self.planner.ik_robot_frame(
            robot_target=tablet_pose_robot_frame
        )

        # check IK accuracy
        q = [v for v in ik_solution.values()]
        fk = self.planner.fk(q)
        fk_position = np.array(fk.translation())
        fk_orientation = np.array(R.from_matrix(fk.rotationMatrix()).as_quat().tolist())

        position_error = np.linalg.norm(tablet_position - fk_position)
        orientation_error = np.linalg.norm(tablet_orientation - fk_orientation)

        if position_error > self.max_ik_error or orientation_error > self.max_ik_error:
            self.get_logger().error(
                "PlanTabletPoseService::plan_tablet_callback: IK error too large!"
            )
            response.success = False
            return response

        # check adherence to joint limits
        out_of_range = False
        replan_arm_extension = False
        replan_lift = False
        replan_yaw = False
        for joint_name, joint_position in ik_solution.items():
            if joint_name in JOINT_LIMITS:
                joint_limits = JOINT_LIMITS[joint_name]
                if joint_position < joint_limits[0] or joint_position > joint_limits[1]:
                    if joint_name == "arm_extension":
                        self.get_logger().warn(
                            "PlanTabletPoseService::plan_tablet_callback: IK solution violates joint limit"
                            " for arm, replanning..."
                        )
                        replan_arm_extension = True
                    elif joint_name == "lift":
                        if joint_position > joint_limits[1]:
                            # too high
                            self.get_logger().warn(
                                "PlanTabletPoseService::plan_tablet_callback: IK solution violates joint limit"
                                " for lift, replanning..."
                            )
                            replan_lift = True
                        else:
                            # too low
                            out_of_range = True
                    elif joint_name == "yaw":
                        replan_yaw = True
                    else:
                        out_of_range = True
                        self.get_logger().error(
                            "PlanTabletPoseService::plan_tablet_callback: IK solution violates joint limit for "
                            + str(joint_name)
                        )

        if out_of_range:
            response.success = False
            return response

        # adjust to point at head if too far away
        if replan_arm_extension:
            ik_solution, _ = self.planner.ik_robot_frame(
                robot_target=human_head_root,
            )
            ik_solution["arm_extension"] = max(
                JOINT_LIMITS["arm_extension"][0],
                min(
                    ik_solution["arm_extension"],
                    JOINT_LIMITS["arm_extension"][1],
                ),
            )
            # fk = self.planner.fk([v for v in ik_solution.values()])
            # fk_position = np.array(fk.translation())
            # position_error_vector = tablet_position - fk_position
            # xy = position_error_vector[:2]
            # theta = np.arctan2(xy[1], xy[0])
            theta = 0.0
            ik_solution["yaw"] = theta

        if replan_lift:
            # set to max lift and pitch up a bit
            ik_solution["lift"] = max(
                JOINT_LIMITS["lift"][0],
                min(
                    ik_solution["lift"],
                    JOINT_LIMITS["lift"][1],
                ),
            )

            ik_solution["pitch"] = 0.325

        if replan_yaw:
            ik_solution["yaw"] = 0.0

        # save response
        response.tablet_pose_robot_frame = generate_pose_stamped(
            tablet_position, tablet_orientation, self.now()
        )
        response.robot_ik_joint_names = [k for k in ik_solution.keys()]
        response.robot_ik_joint_positions = [v for v in ik_solution.values()]
        response.plan_time_s = time.time() - plan_start_time
        response.success = True

        return response


def main(args=None):
    rclpy.init(args=args)
    node = PlanTabletPoseService()
    node.get_logger().info("Plan Tablet Pose Service has been started.")
    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()

import os
import time
from typing import Any, Dict, List, Tuple

# test
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import sophuspy as sp
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from stretch.motion.pinocchio_ik_solver import PinocchioIKSolver

from stretch_show_tablet.human import Human, generate_test_human
from stretch_show_tablet.plot_tools import plot_coordinate_frame
from stretch_show_tablet.utils import Ry, spherical_to_cartesian, vector_projection

EPS = 10.0e-6


class TabletPlanner:
    def __init__(self):
        self.controlled_joints = [
            "joint_mobile_base_rotation",
            "joint_lift",
            "joint_arm_l0",
            # "joint_arm_l1",
            # "joint_arm_l2",
            # "joint_arm_l3",
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "joint_wrist_roll",
        ]

        urdf_path = os.path.join(
            os.path.expanduser("~"),
            "ament_ws/src/stretch_show_tablet/stretch_show_tablet/description/stretch_base_rotation_ik.urdf",
        )
        self.ik_solver = PinocchioIKSolver(
            urdf_path=urdf_path,
            ee_link_name="link_grasp_center",
            # ee_link_name="link_gripper_s3_body",
            controlled_joints=self.controlled_joints,
        )

        self.lower_joint_limits = {
            "base": -3.14,  # rotation
            # "base": -5.,  # translation
            "lift": 0.0,
            "arm_extension": 0.0,
            "yaw": -1.75,
            "pitch": -1.57,
            "roll": -3.14,
        }

        self.upper_joint_limits = {
            "base": 3.14,  # rotation
            # "base": 5.,  # translation
            "lift": 1.1,
            "arm_extension": 0.13 * 4,
            "yaw": 4.0,
            "pitch": 0.56,
            "roll": 3.14,
        }

        self.joint_cost_weights = {
            "base": 1.0,
            "lift": 0.1,
            "arm_extension": 10.0,
            "yaw": 1.0,
            "pitch": 0.1,
            "roll": 0.1,
        }

    @staticmethod
    def _get_shoulder_vector(human: Human) -> np.array:
        """
        Returns the vector pointing from the left shoulder to the right shoulder.

        Args:
            human (Human): human with populated pose estimate.

        Returns:
            np.ndarray: 3x1 vector
        """

        if not human.pose_estimate.is_body_populated():
            raise AttributeError(
                "TabletPlanner::_get_shoulder_vector: human body pose not populated!"
            )

        try:
            l_shoulder = np.array(
                human.pose_estimate.body_estimate_robot_frame["left_shoulder"]
            )
            r_shoulder = np.array(
                human.pose_estimate.body_estimate_robot_frame["right_shoulder"]
            )
        except KeyError as e:
            print("TabletPlanner::_get_shoulder_vector: ", e)
            return None

        if any(np.isnan(l_shoulder)) or any(np.isnan(r_shoulder)):
            raise ValueError(
                "TabletPlanner::_get_shoulder_vector: at least one shoulder is NaN!"
            )

        return r_shoulder - l_shoulder

    @staticmethod
    def _get_head_shoulder_orientation(human: Human) -> np.ndarray:
        """
        Gets orientation of the human based on the human's shoulder points.
        X axis: pointing out of back of head.
        Y axis: pointing from left to right shoulder.
        Z axis: against gravity.

        Args:
            human (Human): human with populated pose estimate.

        Returns:
            np.ndarray: 3x3 rotation matrix corresponding to head coordinate frame
        """

        y = TabletPlanner._get_shoulder_vector(human)
        if y is None:
            raise ValueError(
                "TabletPlanner::_get_head_shoulder_orientation: shoulder vector is None"
            )

        z = np.array([0, 0, 1])
        proj_z_y = vector_projection(z, y)
        y = y - proj_z_y
        x = np.cross(y, z)

        for v in [x, y, z]:
            if np.linalg.norm(v) < EPS:
                raise ValueError(
                    "TabletPlanner::_get_head_shoulder_orientation: head rotation matrix creation failed, "
                    "at least one axis has length 0!"
                )

        # normalize
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = z / np.linalg.norm(z)

        rotation_matrix = np.array([x, y, z]).T

        return rotation_matrix

    @staticmethod
    def in_front_of_eyes(human: Human) -> Tuple[sp.SE3, sp.SE3]:
        """
        Returns the location of the tablet relative to the human.
        Places tablet in front of eyes.
        Human's pose estimate should be in the robot's base frame.

        Args:
            human (Human): human object with populated pose estimate.

        Returns:
            Tuple[sp.SE3]: (tablet pose relative to robot's base, human head pose relative to robot's base)

        Raises:
            KeyError: keypoints missing from pose estimate

        """

        # define position and rotation of the tablet in head frame.
        d = human.preferences["eye_distance"]  # in front of the eyes
        z = human.preferences["eye_height"]  # up from the eyes
        d = -d  # x axis points out of the back of the head
        p = np.array([d, 0.0, z])
        # r = np.eye(3)
        theta = human.preferences["tilt_angle"]
        r = Ry(theta)

        # TODO: add rotate about y by tilt_angle
        # TODO: add rotate about x by +/- 90 if portrait?

        tablet_head_frame = sp.SE3(r, p)

        # get head position
        try:
            human_head_root_robot_frame = human.pose_estimate.body_estimate_robot_frame[
                "nose"
            ]
        except KeyError as e:
            print("TabletPlanner::in_front_of_eyes: " + str(e))
            return None

        # get head orientation
        try:
            r_head = TabletPlanner._get_head_shoulder_orientation(human)
        except Exception as e:
            print("TabletPlanner::in_front_of_eyes: " + str(e))
            return None

        # make SE3 and catch errors (e.g., non-orthogonal R)
        try:
            human_head_root = sp.SE3(r_head, human_head_root_robot_frame)
        except Exception as e:
            print("TabletPlanner::in_front_of_eyes: " + str(e))
            return None

        tablet_robot_frame = human_head_root * tablet_head_frame

        return (tablet_robot_frame, human_head_root)

    @staticmethod
    def compute_tablet_rotation_matrix(
        point: npt.ArrayLike, azimuth: float
    ) -> np.ndarray:
        """
        Helper method to compute the rotation matrix associated with a point on a sphere.
        X points towards the center of the sphere.
        Y attempts to point tangent to the sphere, parallel with the floor.
        Z is the cross product of X and Y.

        Args:
            point (npt.ArrayLike): 3x1 point on the surface of a sphere centered at [0, 0, 0]
            azimuth (float): angle (rad) rotated CW around Z

        Returns:
            np.ndarray: 3x3 rotation matrix
        """

        Rz = np.ndarray(
            [
                [np.cos(azimuth), -np.sin(azimuth), 0],
                [np.sin(azimuth), np.cos(azimuth), 0],
                [0, 0, 1],
            ]
        )

        x = -1 * np.atleast_2d(point).T
        y = Rz @ (np.atleast_2d([0, -1, 0]).T)
        z = np.cross(x.T, y.T).T

        r = np.array([x, y, z]).T
        r = np.squeeze(r)

        r = sp.to_orthogonal_3d(r)

        return r

    @staticmethod
    def generate_tablet_view_points(radius: float = 0.5, n: int = 9) -> List[sp.SE3]:
        """
        Helper function to generate candidate tablet showing points around someone's head.

        Args:
            radius (float): radius from head in meters
            n (int): 6 or 9, number of frames to test. 6 only contains points on sagittal plane and one side of
            the body.

        Returns:
            List[sp.SE3]: list length n of coordinate frames for tablet placements w.r.t. human head
        """

        if n == 9:
            azimuths = [-30.0, 0.0, 30.0]
        elif n == 6:
            azimuths = [0.0, -30.0]
        else:
            raise ValueError(
                "TabletPlanner::generate_tablet_view_points: incorrect n of samples."
            )

        angles = [90.0, 112.5, 135.0]

        azimuths = [np.deg2rad(a) for a in azimuths]
        angles = [np.deg2rad(a) for a in angles]

        frames = []

        for az in azimuths:
            for an in angles:
                point = np.array(spherical_to_cartesian(radius, az, an))
                r = TabletPlanner.compute_tablet_rotation_matrix(point, az)
                frames.append(sp.SE3(r, point))

        return frames

    def cost_midpoint_displacement(self, q: dict) -> float:
        """
        Computes midpoint displacement cost for a joint configuration q

        Args:
            q (dict): joint configuration

        Returns:
            float: cost
        """
        # see (3.57) in Siciliano - midpoint distance cost
        cost = 0.0
        n = len(q.keys())
        for key, value in q.items():
            # compute term
            lo = self.lower_joint_limits[key]
            hi = self.upper_joint_limits[key]
            mid = (hi + lo) / 2.0

            numerator = value - mid
            denominator = hi - lo
            term = (numerator / denominator) ** 2.0

            # add weight
            weight = self.joint_cost_weights[key]

            cost += weight * term

        cost = (1.0 / (2.0 * n)) * cost
        return cost

    def _ik_cost_optimization_target(self, xy, handle_cost_function, world_target):
        """
        Wraps IK solver for use in optimizer.
        TODO: check that this still works after API changes in July 2024

        Args:
            xy: TODO
            handle_cost_function (function handle): function handle for cost function
            world_target: TODO

        Returns:
            list: 3x1 list [x, y, theta] of optimal base location
        """
        # run IK
        r = np.eye(3)
        p = np.array([xy[0], xy[1], 0.0])
        world_base_link = sp.SE3(r, p)
        q, ik_stats = self.ik(
            world_base_link=world_base_link, world_target=world_target
        )

        # gains
        k_f = 1.0
        k_err = 10.0

        # costs
        f_cost = handle_cost_function(q)
        error_cost = np.linalg.norm(ik_stats["final_error"])
        total_cost = k_f * f_cost + k_err * error_cost
        return total_cost

    def get_base_location(self, handle_cost_function, tablet_pose_world: sp.SE3):
        """
        Optimizes robot base location to present a tablet at a point in world coordinates.
        TODO: check that this still works after API changes in July 2024

        Args:
            handle_cost_function (function handle): function handle for cost function
            tablet_pose_world (sp.SE3): tablet pose in world coordinates
        """
        # TODO: add in human pose for removing points near the human

        # heuristics from workspace sampling
        r = 0.5
        th = np.deg2rad(45.0)

        # get initial guess
        base_rotation = np.eye(3)
        base_position = np.array([-r * np.sin(th), r * np.cos(th), 0.0])
        base_pose_tablet = sp.SE3(base_rotation, base_position)
        base_pose_world = tablet_pose_world * base_pose_tablet

        initial_xy = base_pose_world.translation()[:2]

        result = minimize(
            lambda params: self._ik_cost_optimization_target(
                params, handle_cost_function, tablet_pose_world
            ),
            initial_xy,
            method="CG",
        )
        return result.x

    @staticmethod
    def reachable(human: Human):
        raise NotImplementedError

    def fk(self, q_state: npt.ArrayLike) -> sp.SE3:
        """
        Runs FK on robot pose.

        Args:
            q_state (npt.ArrayLike): joint state vector

        Returns:
            sp.SE3: transform from robot base link to end effector
        """

        position, orientation = self.ik_solver.compute_fk(q_state)
        r = R.from_quat(orientation).as_matrix()
        return sp.SE3(r, position)

    def ik_robot_frame(
        self, robot_target: sp.SE3, debug: bool = False
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Runs IK in the robot's base frame.

        Args:
            robot_target (sp.SE3): target in robot base frame
            debug: unused

        Returns:
            dict: joint state IK solution
        """

        # compute IK
        pos_desired = robot_target.translation()
        quat_desired = R.from_matrix(robot_target.rotationMatrix()).as_quat()
        q_soln, success, stats = self.ik_solver.compute_ik(
            pos_desired=pos_desired,
            quat_desired=quat_desired,
        )

        if debug:
            pass

        base_drive = q_soln[0]
        lift = q_soln[1]
        arm_ext = q_soln[2]
        yaw = q_soln[3]
        pitch = q_soln[4]
        roll = q_soln[5]

        result = {
            "base": base_drive,
            "lift": lift,
            "arm_extension": arm_ext,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
        }

        return result, stats

    def ik(
        self,
        world_target: sp.SE3,
        world_base_link: sp.SE3 = sp.SE3(),
        debug: bool = False,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Runs IK in the robot's base frame.
        TODO: check that this still works after API changes in July 2024

        Args:
            world_target (sp.SE3): target in world frame
            world_base_link (sp.SE3): location of robot base in world frame
            debug: whether to check IK error and print to terminal

        Returns:
            dict: joint state IK solution
        """
        # transform target in world frame to base frame
        target_base_frame = world_base_link.inverse() * world_target

        # compute IK
        pos_desired = target_base_frame.translation()
        quat_desired = R.from_matrix(target_base_frame.rotationMatrix()).as_quat()
        q_soln, success, stats = self.ik_solver.compute_ik(
            pos_desired=pos_desired,
            quat_desired=quat_desired,
        )

        if debug:
            fk = self.ik_solver.compute_fk(q_soln)
            # err = np.concatenate([pos_desired, quat_desired]) - np.concatenate([fk[0], fk[1]])
            # print("error:", [f"{e:.4f}" for e in err])
            # print(stats)
            fk_se3 = sp.SE3(R.from_quat(fk[1]).as_matrix(), fk[0])
            fk_world = world_base_link * fk_se3
            print("original target:")
            print(world_target.translation())
            print("fk:")
            print(fk_world.translation())

        base_drive = q_soln[0]
        lift = q_soln[1]
        arm_ext = q_soln[2]
        yaw = q_soln[3]
        pitch = q_soln[4]
        roll = q_soln[5]

        result = {
            "base": base_drive,
            "lift": lift,
            "arm_extension": arm_ext,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
        }

        return result, stats


# tests
def test(args):
    data_dir = args.data_dir
    # _test_spherical_coordinates()
    # _test_cost_function()
    # _test_generate_points(data_dir)
    _test_base_optimizer(data_dir)


def _test_spherical_coordinates():
    f = plt.figure()
    a = f.add_subplot(1, 1, 1, projection="3d")

    frames = TabletPlanner.generate_tablet_view_points()
    for frame in frames:
        plot_coordinate_frame(a, frame.translation(), frame.rotationMatrix(), l=0.1)

    plot_coordinate_frame(a, [0, 0, 0], np.eye(3), l=0.25)

    a.set_xlim([-1.0, 1.0])
    a.set_ylim([-1.0, 1.0])
    a.set_zlim([-1.0, 1.0])
    a.set_xlabel("x (m)")
    a.set_ylabel("y (m)")
    a.set_zlabel("z (m)")
    a.set_aspect("equal")
    plt.show()


def _test_cost_function():
    planner = TabletPlanner()
    target = sp.SE3(np.eye(3), [0.25, -0.3, 0.6])
    q, _ = planner.ik(target)
    planner.cost_midpoint_displacement(q)


def _test_generate_points(data_dir):
    tp = TabletPlanner()

    # for i in range(20):
    for i in [10]:
        human = generate_test_human(data_dir, i)
        tablet = TabletPlanner.in_front_of_eyes(human)
        q_soln, _ = tp.ik(tablet)
        print(q_soln)


def _test_base_optimizer(data_dir):
    tp = TabletPlanner()
    human = generate_test_human(data_dir, 1)
    tablet = TabletPlanner.in_front_of_eyes(human)
    start = time.time()
    xy = tp.get_base_location(tp.cost_midpoint_displacement, tablet)
    duration = time.time() - start
    p = np.array([xy[0], xy[1], 0.0])
    r = np.eye(3)
    print("Result computed in", duration, "seconds")
    print(sp.SE3(r, p))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)

    args = parser.parse_args()
    test(args)

import numpy as np
import sophuspy as sp
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

from stretch.motion.pinocchio_ik_solver import PinocchioIKSolver

from stretch_tablet.human import Human, generate_test_human
from stretch_tablet.utils import spherical_to_cartesian

import os
import time

# test
import matplotlib.pyplot as plt
from stretch_tablet.plot_tools import plot_coordinate_frame

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
            "joint_wrist_roll"
        ]

        urdf_path = os.path.join(os.path.expanduser("~"), "ament_ws/src/stretch_show_tablet/stretch_tablet/description/stretch_base_rotation_ik.urdf")
        self.ik_solver = PinocchioIKSolver(
            urdf_path=urdf_path,
            ee_link_name="link_grasp_center",
            # ee_link_name="link_gripper_s3_body",
            controlled_joints=self.controlled_joints
        )

        self.lower_joint_limits = {
            "base": -3.14,  # rotation
            # "base": -5.,  # translation
            "lift": 0.,
            "arm_extension": 0.,
            "yaw": -1.75,
            "pitch": -1.57,
            "roll": -3.14,
        }

        self.upper_joint_limits = {
            "base": 3.14,  # rotation
            # "base": 5.,  # translation
            "lift": 1.1,
            "arm_extension": 0.13 * 4,
            "yaw": 4.,
            "pitch": 0.56,
            "roll": 3.14,
        }

        self.joint_cost_weights = {
            "base": 1.,
            "lift": 0.1,
            "arm_extension": 10.,
            "yaw": 1.,
            "pitch": 0.1,
            "roll": 0.1,
        }

    @staticmethod
    def in_front_of_eyes(human: Human) -> sp.SE3:
        """
        Relative to eyes / forehead
        Returns in world frame
        """

        d = human.preferences["eye_distance"]
        p = np.array([d, 0., 0.])
        r = np.diag([-1., -1., 1.])  # Rotate Z by 180*
        # TODO: add rotate about y by tilt_angle
        # TODO: add rotate about x by +/- 90 if portrait?

        tablet = sp.SE3(r, p)

        try:
            human_head_root = human.pose_estimate.body_estimate["nose"]
        except KeyError as e:
            print("TabletPlanner::in_front_of_eyes: " + str(e))
            return None
            
        human_head_root_world = human.pose_estimate.get_point_world(human_head_root)
        r_head = [[0,-1,0], [1,0,0], [0,0,1]]
        human_head_root = sp.SE3(r_head, human_head_root_world)

        tablet_world = human_head_root * tablet

        return tablet_world
    
    @staticmethod
    def compute_tablet_rotation_matrix(point, azimuth):
        Rz = np.array([[np.cos(azimuth), -np.sin(azimuth), 0], [np.sin(azimuth), np.cos(azimuth), 0], [0, 0, 1]])

        x = -1 * np.atleast_2d(point).T
        y = Rz @ (np.atleast_2d([0, -1, 0]).T)
        z = np.cross(x.T, y.T).T

        r = np.array([x, y, z]).T
        r = np.squeeze(r)

        r = sp.to_orthogonal_3d(r)

        return r

    @staticmethod
    def generate_tablet_view_points(radius=0.5, n=9):
        """
        generates points in the head frame
        """
        if n == 9:
            azimuths = [-30., 0., 30.]
        elif n == 6:
            azimuths = [0., -30.]
        else:
            raise ValueError
        
        angles = [90., 112.5, 135.]

        azimuths = [np.deg2rad(a) for a in azimuths]
        angles = [np.deg2rad(a) for a in angles]

        frames = []

        for az in azimuths:
            for an in angles:
                point = np.array(spherical_to_cartesian(radius, az, an))
                r = TabletPlanner.compute_tablet_rotation_matrix(point, az)
                frames.append(sp.SE3(r, point))
        
        return frames
    
    def cost_midpoint_displacement(self, q):
        # see (3.57) in Siciliano - midpoint distance cost
        cost = 0.
        n = len(q.keys())
        for key, value in q.items():
            # compute term
            lo = self.lower_joint_limits[key]
            hi = self.upper_joint_limits[key]
            mid = (hi + lo) / 2.

            numerator = value - mid
            denominator = hi - lo
            term = (numerator / denominator) ** 2.

            # add weight
            weight = self.joint_cost_weights[key]
            
            cost += weight * term

        cost = (1. / (2. * n)) * cost
        return cost

    def _ik_cost_optimization_target(self, xy, handle_cost_function, world_target):
        # run IK
        r = np.eye(3)
        p = np.array([xy[0], xy[1], 0.])
        world_base_link = sp.SE3(r, p)
        q, ik_stats = self.ik(world_base_link=world_base_link, world_target=world_target)
        
        # gains
        k_f = 1.
        k_err = 10.

        # costs
        f_cost = handle_cost_function(q)
        error_cost = np.linalg.norm(ik_stats["final_error"])
        total_cost = k_f * f_cost + k_err * error_cost
        return total_cost

    def get_base_location(self, handle_cost_function, tablet_pose_world: sp.SE3):
        # TODO: add in human pose for removing points near the human

        # heuristics from workspace sampling
        r = 0.5
        th = np.deg2rad(45.)

        # get initial guess
        base_rotation = np.eye(3)
        base_position = np.array(
            [
                -r * np.sin(th),
                r * np.cos(th),
                0.
            ]
        )
        base_pose_tablet = sp.SE3(base_rotation, base_position)
        base_pose_world = tablet_pose_world * base_pose_tablet
        
        initial_xy = base_pose_world.translation()[:2]
        
        result = minimize(lambda params: self._ik_cost_optimization_target(params, handle_cost_function, tablet_pose_world), initial_xy, method='CG')
        return result.x

    @staticmethod
    def reachable(human: Human):
        raise NotImplementedError

    def fk(self, q_state) -> sp.SE3:
        position, orientation = self.ik_solver.compute_fk(q_state)
        r = R.from_quat(orientation).as_matrix()
        return sp.SE3(r, position)

    def ik(self, world_target: sp.SE3, world_base_link: sp.SE3 = sp.SE3(), debug: bool=False):
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
            print('original target:')
            print(world_target.translation())
            print('fk:')
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
    a = f.add_subplot(1, 1, 1, projection='3d')

    frames = TabletPlanner.generate_tablet_view_points()
    for frame in frames:
        plot_coordinate_frame(a, frame.translation(), frame.rotationMatrix(), l=0.1)
    
    plot_coordinate_frame(a, [0,0,0], np.eye(3), l=0.25)

    a.set_xlim([-1., 1.])
    a.set_ylim([-1., 1.])
    a.set_zlim([-1., 1.])
    a.set_xlabel('x (m)')
    a.set_ylabel('y (m)')
    a.set_zlabel('z (m)')
    a.set_aspect('equal')
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
    p = np.array([xy[0], xy[1], 0.])
    r = np.eye(3)
    print("Result computed in", duration, "seconds")
    print(sp.SE3(r, p))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)

    args = parser.parse_args()
    test(args)

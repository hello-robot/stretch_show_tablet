import numpy as np
import sophuspy as sp
from scipy.spatial.transform import Rotation as R

from stretch.motion.pinocchio_ik_solver import PinocchioIKSolver

import os
import json
from enum import Enum

# testing
import matplotlib.pyplot as plt
from plot_tools import plot_coordinate_frame

EPS = 10.e-9

# helper functions
def load_bad_json_data(data_string):
    data_string = data_string.replace("'", "\"")
    data_string = data_string.replace("(", "[")
    data_string = data_string.replace(")", "]")

    data = json.loads(data_string)
    return data

def load_bad_json(file):
    with open(file) as f:
        data_string = json.load(f)
        return load_bad_json_data(data_string)

landmark_names = ['nose', 'neck',
                'right_shoulder', 'right_elbow', 'right_wrist',
                'left_shoulder', 'left_elbow', 'left_wrist',
                'right_hip', 'right_knee', 'right_ankle',
                'left_hip', 'left_knee', 'left_ankle',
                'right_eye', 'left_eye',
                'right_ear', 'left_ear']

def spherical_to_cartesian(radius, azimuth, elevation):
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        radius (float): The radius or radial distance.
        azimuth (float): The azimuth angle in radians.
        elevation (float): The elevation angle in radians.
    
    Returns:
        list: A list containing the Cartesian coordinates (x, y, z).
    """
    x = radius * np.sin(elevation) * np.cos(azimuth)
    y = radius * np.sin(elevation) * np.sin(azimuth)
    z = radius * np.cos(elevation)
    
    return [x, y, z]

def in_range(value, range):
    return True if value >= range[0] and value <= range[1] else False

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    IN = 5
    OUT = 6

def get_vector_direction_image_plane(v: np.ndarray) -> Direction:
    x = v[0]
    y = v[1]
    
    theta = np.arctan2(y, x)

    PI_4 = np.pi / 4.
    if in_range(theta, [-PI_4, PI_4]):
        return Direction.RIGHT
    elif in_range(theta, [-3*PI_4, -PI_4]):
        return Direction.DOWN
    elif in_range(theta, [PI_4, 3*PI_4]):
        return Direction.UP
    else:
        return Direction.LEFT

# objects
class HumanKinematics:
    def __init__(self):
        # transforms
        self.root = sp.SE3()
        self.root2l_shoulder_pose = sp.SE3()
        self.root2r_shoulder_pose = sp.SE3()
        self.root2head_pose = sp.SE3()
        self.head2l_eye_pose = sp.SE3()
        self.head2r_eye_pose = sp.SE3()
        self.head_vision_vector = np.array([1., 0., 0.])

class HumanPoseEstimate:
    def __init__(self):
        self.world2camera_pose = sp.SE3()
        self.body_estimate = None
        self.body_points = None
        self.face_estimate = None
        self.face_points = None

    def load_face_estimate(self, file):
        data = load_bad_json(file)
        self.set_face_estimate(data)

    def set_face_estimate(self, data):
        self.face_estimate = data
        self.face_points = np.array([v for v in data.values()]).T

    def load_body_estimate(self, file):
        data = load_bad_json(file)
        self.set_body_estimate(data)

        not_visible = landmark_names.copy()
        for key, value in data.items():
            if np.linalg.norm(value) > EPS:
                not_visible.remove(key)

        print("Cannot see: " + str(not_visible))

    def set_body_estimate(self, data):
        self.body_estimate = data
        self.body_points = np.array([v for v in data.values()]).T

    def load_camera_pose(self, camera_file):
        with open(camera_file) as f:
            data = json.load(f)

        position = np.array(data[0])
        quaternion = np.array(data[1])

        rotation_matrix = R.from_quat(quaternion).as_matrix()

        # NOTE: this is backwards bc test code is wrong
        camera2world_pose = sp.SE3(rotation_matrix, position.T)
        self.set_camera_pose(camera2world_pose.inverse())

    def set_camera_pose(self, world2camera_pose: sp.SE3):
        self.world2camera_pose = world2camera_pose

    def clear_estimates(self):
        self.body_estimate = None
        self.body_points = None
        self.face_estimate = None
        self.face_points = None

    def is_populated(self):
        return True if self.body_estimate is not None and self.body_points is not None \
            and self.face_estimate is not None and self.face_points is not None else False

    def get_point_world(self, point):
        point = np.array(point).T
    
        point_world = sp.transform_points_by_poses(self.world2camera_pose.matrix3x4().ravel(), point).T
        return point_world

    def get_face_world(self):
        if self.face_points is None:
            raise ValueError
        
        world_points = sp.transform_points_by_poses(self.world2camera_pose.matrix3x4().ravel(), self.face_points.T).T
        return world_points
    
    def get_body_world(self):
        if self.body_points is None:
            raise ValueError
        
        world_points = sp.transform_points_by_poses(self.world2camera_pose.matrix3x4().ravel(), self.body_points.T).T
        return world_points

class Human:
    def __init__(self) -> None:
        self.kinematics = HumanKinematics()
        self.pose_estimate = HumanPoseEstimate()
        self.preferences = self.init_preferences()

    def init_preferences(self):
        p = {
            "eye_distance": 0.5,  # 50cm, or ~19in
            "portait": True,  # whether to use portrait instead of landscape
            "tilt_angle": 0.  # radians
        }

        return p
    
    def update_preferences(self, new_preferences: dict):
        for key, value in new_preferences.items():
            self.preferences[key] = value

class TabletController:
    def __init__(self):
        self.tablet_horizontal_deadband = [-0.1, 0.1]
        self.tablet_portrait_x_offset = 0.15

    def get_head_vertical_vector(self, human: Human):
        face_landmarks = human.pose_estimate.face_estimate
        chin_xyz = np.array(face_landmarks["chin_middle"])
        nose_xyz = np.array(face_landmarks["nose_tip"])
        return nose_xyz - chin_xyz

    def get_head_direction(self, human: Human):
        face_vector = self.get_head_vertical_vector(human)
        return get_vector_direction_image_plane(face_vector)

    def get_tablet_yaw_action(self, human: Human):
        """
        Returns rad
        """
        face_landmarks = human.pose_estimate.face_estimate
        chin_xyz = face_landmarks["chin_middle"]

        head_direction = self.get_head_direction(human)

        if head_direction == Direction.UP:
            x = chin_xyz[0]
        elif head_direction == Direction.LEFT:
            x = chin_xyz[1] + self.tablet_portrait_x_offset
        elif head_direction == Direction.RIGHT:
            x = -chin_xyz[1] - self.tablet_portrait_x_offset
        else:
            return 0.

        if in_range(x, self.tablet_horizontal_deadband):
            return 0.
        
        Kp = 0.2
        yaw_action = Kp * (-1 * x)
        return yaw_action

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

        urdf_path = os.path.join(os.path.expanduser("~"), "ament_ws/src/stretch_tablet/description/stretch_base_rotation_ik.urdf")
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

        human_head_root = human.pose_estimate.body_estimate["nose"]
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

    @staticmethod
    def reachable(human: Human):
        pass

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

def generate_test_human(data_dir, i=6):
    body_path = data_dir + "body_" + str(i) + ".json"
    face_path = data_dir + "face_" + str(i) + ".json"
    camera_path = data_dir + "camera_" + str(i) + ".json"

    human = Human()
    human.pose_estimate.load_face_estimate(face_path)
    human.pose_estimate.load_body_estimate(body_path)
    human.pose_estimate.load_camera_pose(camera_path)
    return human

def test_spherical_coordinates():
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

def test_cost_function():
    planner = TabletPlanner()
    planner.cost_midpoint_displacement(None)

def main(args):
    # test_spherical_coordinates()
    test_cost_function()
    return

    tp = TabletPlanner()

    # for i in range(20):
    for i in [10]:
        human = generate_test_human(args.data_dir, i)
        tablet = TabletPlanner.in_front_of_eyes(human)
        q_soln, _ = tp.ik(tablet)
        print(q_soln)
        return
        
        f = plt.figure()
        a = f.add_subplot(1, 1, 1, projection='3d')
        a.scatter(*human.pose_estimate.get_face_world())
        a.scatter(*human.pose_estimate.get_body_world())

        plot_coordinate_frame(a, tablet.translation(), tablet.rotationMatrix(), l=0.1)

        a.set_xlabel('x (m)')
        a.set_ylabel('y (m)')
        a.set_zlabel('z (m)')
        
        # a.set_xlim([-2, 2])
        # a.set_ylim([-2, 2])
        # a.set_zlim([0, 2])

        # a.set_xlim([1, 2])
        # a.set_ylim([-.4, .4])
        # a.set_zlim([0, 1.2])

        a.set_xlim([-0.5, 0.5])
        a.set_ylim([-2., -1.])
        a.set_zlim([0., 2.])

        a.set_aspect('equal')

        plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)

    args = parser.parse_args()
    main(args)
import numpy as np
import sophuspy as sp
from scipy.spatial.transform import Rotation as R

from stretch.motion.pinocchio_ik_solver import PinocchioIKSolver

import json

EPS = 10.e-9

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

        self.world2camera_pose = sp.SE3(rotation_matrix, position.T)
        self.world2camera_pose = self.world2camera_pose.inverse()

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

class TabletPlanner:
    def __init__(self):
        self.controlled_joints = [
            "x_prismatic_joint",
            "joint_lift",
            "joint_arm_l0",
            "joint_arm_l1",
            "joint_arm_l2",
            "joint_arm_l3",
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "joint_wrist_roll"
        ]
        self.ik_solver = PinocchioIKSolver(
            urdf_path="/home/hello-robot/ament_ws/src/stretch_tablet/description/stretch_re3.urdf",
            ee_link_name="link_grasp_center",
            controlled_joints=self.controlled_joints
        )

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
    def reachable(human: Human):
        pass

    def ik(self, world_target: sp.SE3, world_base_link: sp.SE3 = sp.SE3()):
        # TODO: use world_base_link to transform the target into base link frame
        target_base_frame = world_base_link.inverse() * world_target
        pos_desired = target_base_frame.translation()
        quat_desired = R.from_matrix(target_base_frame.rotationMatrix()).as_quat()
        q_soln, success, stats = self.ik_solver.compute_ik(
            pos_desired=pos_desired,
            quat_desired=quat_desired,
        )

        base_drive = q_soln[0]
        lift = q_soln[1]
        arm_ext = sum(q_soln[2:5])
        yaw = q_soln[6]
        pitch = q_soln[7]
        roll = q_soln[8]
        
        result = {
            "base": base_drive,
            "lift": lift,
            "arm_extension": arm_ext,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
        }

        return result

def generate_test_human(data_dir, i=6):
    body_path = data_dir + "body_" + str(i) + ".json"
    face_path = data_dir + "face_" + str(i) + ".json"
    camera_path = data_dir + "camera_" + str(i) + ".json"

    human = Human()
    human.pose_estimate.load_face_estimate(face_path)
    human.pose_estimate.load_body_estimate(body_path)
    human.pose_estimate.load_camera_pose(camera_path)
    return human

def main(args):
    import matplotlib.pyplot as plt
    from plot_tools import plot_coordinate_frame

    tp = TabletPlanner()

    # for i in range(20):
    for i in [10]:
        human = generate_test_human(args.data_dir, i)
        tablet = TabletPlanner.in_front_of_eyes(human)
        q_soln = tp.ik(tablet)
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
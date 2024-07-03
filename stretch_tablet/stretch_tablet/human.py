import numpy as np
import sophuspy as sp
from scipy.spatial.transform import Rotation as R
from stretch_tablet.utils import load_bad_json, landmark_names

import json

# testing
import matplotlib.pyplot as plt

EPS = 10.e-9

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

def generate_test_human(data_dir, i=6):
    body_path = data_dir + "body_" + str(i) + ".json"
    face_path = data_dir + "face_" + str(i) + ".json"
    camera_path = data_dir + "camera_" + str(i) + ".json"

    human = Human()
    human.pose_estimate.load_face_estimate(face_path)
    human.pose_estimate.load_body_estimate(body_path)
    human.pose_estimate.load_camera_pose(camera_path)
    return human

# test
def test(args):
    _test_plot_human(args.data_dir)

def _test_plot_human(data_dir):
    human = generate_test_human(data_dir, i=1)
        
    f = plt.figure()
    a = f.add_subplot(1, 1, 1, projection='3d')
    a.scatter(*human.pose_estimate.get_face_world())
    a.scatter(*human.pose_estimate.get_body_world())

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
    test(args)

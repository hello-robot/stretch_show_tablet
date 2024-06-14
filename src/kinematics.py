import numpy as np
import sophuspy as sp
from scipy.spatial.transform import Rotation as R
import json

def pose2SE3(pose) -> sp.SE3:
    pos = pose["position"]
    ori = pose["orientation"]

    position = np.array([pos["x"], pos["y"], pos["z"]])
    orientation = np.array([ori["x"], ori["y"], ori["z"], ori["w"]])

    r = R.from_quat(orientation).as_matrix()
    return sp.SE3(r, position)

def points2np(points: list) -> np.ndarray:
    points_np = np.zeros([3, len(points)])
    for i in range(len(points)):
        point = points[i]
        points_np[:, i] = [point["x"], point["y"], point["z"]]

    return points_np

# def points2SE3(points):
#     points_np = points2np(points)
#     points_se3 = np.array([sp.SE3(np.eye(3), points_np[:, i]) for i in range(points_np.shape[1])])
#     return points_se3

def load_bad_json(file):
    with open(file) as f:
        data_string = json.load(f)
        data_string = data_string.replace("'", "\"")
        data_string = data_string.replace("(", "[")
        data_string = data_string.replace(")", "]")
        data = json.loads(data_string)
    return data

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
        self.camera2body_pose = sp.SE3()
        self.camera2face_pose = sp.SE3()
        self.body_points = None
        self.face_points = None

    def load_face_estimate(self, file):
        data = load_bad_json(file)
        self.face_points = np.array([v for v in data.values()]).T

    def load_body_estimate(self, file):
        data = load_bad_json(file)
        self.body_points = np.array([v for v in data.values()]).T

        # print("Can see:")
        # print([landmark_names[i] + str(self.body_points[:, i]) for i in range(len(landmark_names)) if can_see[i]])
        # print("Cannot see:")
        # print([landmark_names[i] for i in range(len(can_see)) if not can_see[i]])

    def get_face_world(self, world2camera_pose: sp.SE3):
        if self.face_points is None:
            raise ValueError
        
        world_points = sp.transform_points_by_poses(world2camera_pose.matrix3x4().ravel(), self.face_points.T).T
        return world_points
    
    def get_body_world(self, world2camera_pose: sp.SE3):
        if self.body_points is None:
            raise ValueError
        
        world_points = sp.transform_points_by_poses(world2camera_pose.matrix3x4().ravel(), self.body_points.T).T
        return world_points

class Human:
    def __init__(self) -> None:
        self.kinematics = HumanKinematics()
        self.pose_estimate = HumanPoseEstimate()

def generate_test_human(data_dir):
    body_path = data_dir + "body_1.json"
    face_path = data_dir + "face_1.json"

    human = Human()
    human.pose_estimate.load_face_estimate(face_path)
    human.pose_estimate.load_body_estimate(body_path)
    return human

def main(args):
    test_camera_pose = sp.SE3(np.array([[0,0,1],[0,1,0],[-1,0,0]]), np.array([0, 0, 1]))
    human = generate_test_human(args.data_dir)

    import matplotlib.pyplot as plt
    f = plt.figure()
    a = f.add_subplot(1, 1, 1, projection='3d')
    a.scatter(*human.pose_estimate.get_face_world(test_camera_pose))
    a.scatter(*human.pose_estimate.get_body_world(test_camera_pose))
    a.set_xlabel('x (m)')
    a.set_ylabel('y (m)')
    a.set_zlabel('z (m)')
    
    # a.set_xlim([-2, 2])
    # a.set_ylim([-2, 2])
    # a.set_zlim([0, 2])

    a.set_xlim([1, 2])
    a.set_ylim([-.4, .4])
    a.set_zlim([0, 1.2])
    a.set_aspect('equal')

    plt.show()
    # human = Human()
    # print(human.kinematics.root)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)

    args = parser.parse_args()
    main(args)
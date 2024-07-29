from __future__ import annotations

import json
from typing import List, Optional

# testing
import matplotlib.pyplot as plt
import numpy as np
import sophuspy as sp

from stretch_show_tablet.utils import landmark_names, load_bad_json

EPS = 10.0e-9


# helpers
def transform_estimate_dict(estimate: dict, camera_pose: sp.SE3):
    """
    Transforms points contained in a dictionary {point_name: [x, y, z]} by the pose of the camera.
    """
    points = np.array([v for v in estimate.values()])
    transformed_points = sp.transform_points_by_poses(
        camera_pose.matrix3x4().ravel(), points
    )
    transformed_estimate = {k: v for (k, v) in zip(estimate.keys(), transformed_points)}
    return transformed_estimate


# objects
class HumanKinematics:
    def __init__(self):
        """
        TODO: use this for something useful...
        """
        # transforms
        self.root = sp.SE3()
        self.root2l_shoulder_pose = sp.SE3()
        self.root2r_shoulder_pose = sp.SE3()
        self.root2head_pose = sp.SE3()
        self.head2l_eye_pose = sp.SE3()
        self.head2r_eye_pose = sp.SE3()
        self.head_vision_vector = np.array([1.0, 0.0, 0.0])


class HumanPoseEstimate:
    def __init__(self):
        self.body_estimate_robot_frame = None
        self.face_estimate_robot_frame = None
        self.face_estimate_camera_frame = None

    def load_face_estimate(self, file):
        """
        Helper function for loading a human face estimate saved by record_test_data.py
        TODO: refactor!
        """
        raise NotImplementedError
        data = load_bad_json(file)
        self.set_face_estimate(data)

    def set_face_estimate_robot_frame(self, face_data: dict):
        self.face_estimate_robot_frame = face_data

    def set_face_estimate_camera_frame(self, face_data: dict):
        """
        Sets face estimate.

        Args:
            face_data (dict): face points in camera frame {point_name: position}
            camera_pose (sp.SE3): camera pose relative to robot's base_link
        """
        self.face_estimate_camera_frame = face_data
        # self.face_estimate_robot_frame = transform_estimate_dict(face_data, camera_pose)

    def get_face_estimate_camera_frame(self) -> dict:
        return self.face_estimate_camera_frame

    def load_body_estimate(self, file):
        """
        Helper function for loading a human pose estimate saved by record_test_data.py
        TODO: refactor!
        """
        raise NotImplementedError
        data = load_bad_json(file)
        self.set_body_estimate(data)

        not_visible = landmark_names.copy()
        for key, value in data.items():
            if np.linalg.norm(value) > EPS:
                not_visible.remove(key)

        print("Cannot see: " + str(not_visible))

    def set_body_estimate_robot_frame(self, body_data: dict):
        self.body_estimate_robot_frame = body_data

    def get_body_estimate_robot_frame(self):
        return self.body_estimate_robot_frame

    def set_body_estimate_camera_frame(self, body_data: dict, camera_pose: sp.SE3):
        """
        Sets body estimate.

        Args:
            body_data (dict): body points in camera frame {point_name: position}
            camera_pose (sp.SE3): camera pose relative to robot's base_link
        """

        self.body_estimate_robot_frame = transform_estimate_dict(body_data, camera_pose)

    def get_body_estimate_string(self) -> Optional[str]:
        """
        Dumps the current body pose estimate in robot frame to a string.
        """
        if self.body_estimate_robot_frame is not None:
            return json.dumps(self.body_estimate_robot_frame)
        else:
            return None

    def get_body_points(self) -> np.ndarray:
        """
        Returns body poes points in robot frame.
        """
        if self.body_estimate_robot_frame is not None:
            return np.array([v for v in self.body_estimate_robot_frame.values()])

    def clear_estimates(self):
        """
        Helper to clear the state of the object
        """
        self.body_estimate_robot_frame = None
        self.face_estimate_robot_frame = None

    def is_body_populated(self):
        return self.body_estimate_robot_frame is not None

    def is_face_populated(self):
        return self.face_estimate_robot_frame is not None

    def is_populated(self):
        self.is_body_populated() and self.is_face_populated()

    # def get_point_world(self, point):
    #     point = np.array(point).T

    #     point_world = sp.transform_points_by_poses(self.robot2camera_pose.matrix3x4().ravel(), point).T
    #     return point_world

    # def get_face_world(self):
    #     if self.face_points is None:
    #         raise ValueError

    #     world_points = sp.transform_points_by_poses(self.robot2camera_pose.matrix3x4().ravel(), self.face_points.T).T
    #     return world_points

    # def get_body_world(self):
    #     if self.body_points is None:
    #         raise ValueError

    #     world_points = sp.transform_points_by_poses(self.robot2camera_pose.matrix3x4().ravel(), self.body_points.T).T
    # return world_points

    @staticmethod
    def average_pose_estimates(
        pose_estimates: List[HumanPoseEstimate],
    ) -> HumanPoseEstimate:
        # get pose keys
        unique_keys: List[str] = []
        for p in pose_estimates:
            pose = p.get_body_estimate_robot_frame()
            unique_keys = unique_keys + [k for k in pose.keys()]
            unique_keys = list(set(unique_keys))

        # concatenate pose estimates into one dict
        all_poses: dict = {key: [] for key in unique_keys}
        for p in pose_estimates:
            pose = p.get_body_estimate_robot_frame()
            for key in pose.keys():
                all_poses[key].append(pose[key])

        # average pose estimates
        average_pose_estimate_dict = {
            key: np.mean(all_poses[key], axis=0).tolist() for key in unique_keys
        }

        average_pose_estimate = HumanPoseEstimate()
        average_pose_estimate.set_body_estimate_robot_frame(average_pose_estimate_dict)

        return average_pose_estimate


class Human:
    def __init__(self) -> None:
        self.kinematics = HumanKinematics()
        self.pose_estimate = HumanPoseEstimate()
        self.preferences = self.init_preferences()

    def init_preferences(self):
        p = {
            # "eye_distance": 0.5,  # 50cm, or ~19in
            "eye_distance": 0.3,  # 40cm, or ~15.75in
            "eye_height": -0.1,  # 10cm, or ~4in down from nose
            "portrait": False,  # whether to use portrait instead of landscape
            "tilt_angle": np.deg2rad(-10.0),  # radians (positive is down)
        }

        return p

    def update_preferences(self, new_preferences: dict):
        for key, value in new_preferences.items():
            self.preferences[key] = value


def generate_test_human(data_dir, i=6):
    raise NotImplementedError
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
    a = f.add_subplot(1, 1, 1, projection="3d")
    a.scatter(*human.pose_estimate.get_face_world())
    a.scatter(*human.pose_estimate.get_body_world())

    a.set_xlabel("x (m)")
    a.set_ylabel("y (m)")
    a.set_zlabel("z (m)")

    # a.set_xlim([-2, 2])
    # a.set_ylim([-2, 2])
    # a.set_zlim([0, 2])

    # a.set_xlim([1, 2])
    # a.set_ylim([-.4, .4])
    # a.set_zlim([0, 1.2])

    a.set_xlim([-0.5, 0.5])
    a.set_ylim([-2.0, -1.0])
    a.set_zlim([0.0, 2.0])

    a.set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)

    args = parser.parse_args()
    test(args)

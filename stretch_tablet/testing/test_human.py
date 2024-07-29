import numpy as np
import sophuspy as sp

from stretch_tablet.human import Human


def test_setters():
    h = Human()
    test_dict = {"point_" + str(i): [0.2 * i, 0.05 * i, 0.1 * i] for i in range(10)}
    camera_pose = sp.SE3(
        np.diag([-1.0, -1.0, 1.0]), np.array([0.1, -0.1, 1.1])
    )  # Rz by 180, somewhere above base
    h.pose_estimate.set_body_estimate(test_dict, camera_pose)


def main():
    test_setters()


if __name__ == "__main__":
    main()

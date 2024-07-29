import json

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import sophuspy as sp
from matplotlib.patches import Circle
from plot_tools import plot_base_reachability

from .human import generate_test_human
from .planner import TabletPlanner


def main(json_path: str):
    # load human
    human = generate_test_human("/home/lamsey/ament_ws/src/stretch_tablet/data/amy/")
    if json_path[0] == "6":
        n = 6
    elif json_path[0] == "9":
        n = 9
    else:
        raise ValueError

    targets = TabletPlanner.generate_tablet_view_points(n=n)

    # targets to human head frame
    human_head_root = human.pose_estimate.body_estimate["nose"]
    human_head_root_world = human.pose_estimate.get_point_world(human_head_root)
    human_head_pose = sp.SE3([[0, -1, 0], [1, 0, 0], [0, 0, 1]], human_head_root_world)
    targets = [human_head_pose * target for target in targets]

    # get reachable space by robot
    with open(json_path) as f:
        data = json.load(f)

    base_x = np.array(data["base_x"])
    base_y = np.array(data["base_y"])
    counts = np.array(data["counts"])

    # human modeling
    human_points = human.pose_estimate.get_body_world()
    human_points = human_points[:, np.linalg.norm(human_points[:2], axis=0) > 0.1]
    human_xy = human_points[:2, :]

    # plotting
    f = plt.figure()
    a = f.add_subplot(1, 1, 1, projection="3d")  # type: ignore[attr-defined]

    plot_base_reachability(
        a, base_x=base_x, base_y=base_y, counts=counts, targets=targets
    )
    a.scatter(*human_points, color="k", alpha=1)

    human_inflation_distance = 0.5
    for point in human_xy.T:
        # ignore unseen points at [0, 0, 0]
        # print(point)
        # if np.linalg.norm(point) < 0.1:
        #     continue
        p = Circle(
            point,
            radius=human_inflation_distance,
            color="grey",
            alpha=0.3,
            zorder=0,
        )
        a.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
        # Circle(point, 0.1)

    # cfg
    a.set_xlabel("x (m)")
    a.set_ylabel("y (m)")
    a.set_zlabel("z (m)")
    a.view_init(35, 65)
    a.set_aspect("equal")

    plt.show()

    # print(counts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str)

    args = parser.parse_args()
    main(args.json_path)

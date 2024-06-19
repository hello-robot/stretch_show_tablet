import numpy as np
import sophuspy as sp
from scipy.spatial.transform import Rotation as R

from kinematics import TabletPlanner, generate_test_human
from plot_tools import plot_coordinate_frame

PI_2 = np.pi / 2.

def is_valid(q):
    if q["lift"] < 0.1 or q["lift"] > 1.2:
        return False
    if q["arm_extension"] < 0.01 or q["arm_extension"] > 0.5:
        return False
    if q["yaw"] < np.deg2rad(-75.) or q["yaw"] > np.pi:
        return False
    if q["pitch"] < -PI_2 or q["pitch"] > np.deg2rad(30.):
        return False
    if np.abs(q["roll"]) > np.deg2rad(120.):
        return False

    return True

def get_error(stats_dict: dict):
    error = stats_dict["final_error"]
    pos_error = error[:3]
    ori_error = error[3:]
    pos_error_norm = np.linalg.norm(pos_error)
    ori_error_norm = np.linalg.norm(ori_error)
    return pos_error_norm, ori_error_norm

def characterize_tablet_workspace():
    planner = TabletPlanner()
    human = generate_test_human("/home/lamsey/ament_ws/src/stretch_tablet/data/matt/")
    target = planner.in_front_of_eyes(human)

    n = 40
    test_base_x = np.linspace(-1, 1., n)
    test_base_y = np.linspace(-1.5, 0.5, n)
    
    reachable = []
    unreachable = []

    for i in range(n):
        x = test_base_x[i]
        for j in range(n):
            y = test_base_y[j]
            base = sp.SE3(np.eye(3), np.array([x, y, 0]))
            ik_soln, stats = planner.ik(target, base)
            # print(ik_soln)
            pos_error, ori_error = get_error(stats)
            valid = is_valid(ik_soln) and pos_error < 10e-3 and ori_error < 10e-3
            # print(x, y, reachable[i, j])

            if valid:
                reachable.append([x, y])
            else:
                unreachable.append([x, y])

    reachable = np.array(reachable).T
    unreachable = np.array(unreachable).T
    # target_xy = target.translation()[:2]
    target_pos = target.translation()

    import matplotlib.pyplot as plt
    f = plt.figure()
    a = f.add_subplot(1, 1, 1, projection='3d')

    reachable_color = (143. / 255., 235. / 255., 143. / 255.)
    unreachable_color = (240. / 255., 150. / 255., 150. / 255.)

    a.scatter(*reachable, 0, color=reachable_color)
    a.scatter(*unreachable, 0, color=unreachable_color)
    # plt.scatter(*target_pos, s=[100], color='k', marker='x')

    human_pose = human.pose_estimate.get_body_world().T
    human_pose = [p for p in human_pose if np.linalg.norm(p[:2]) > 0.1]
    human_pose = np.array(human_pose).T
    
    a.scatter(*human_pose, color='k')

    a.scatter(*human.pose_estimate.get_face_world(), color='k')

    plot_coordinate_frame(a, target.translation(), target.rotationMatrix(), l=0.2)

    a.set_xlabel('x (m)')
    a.set_ylabel('y (m)')
    a.set_zlabel('z (m)')

    a.set_aspect('equal')

    plt.show()

def test_ik_checker():
    planner = TabletPlanner()

    base_rotation = np.eye(3)
    base_position = np.array([0, 0.2, 0.])
    base = sp.SE3(base_rotation, base_position)

    for y in np.linspace(0.2, -1.2, 10):
        target = sp.SE3(np.eye(3), [0.5, y, 0.75])
        ik_soln, stats = planner.ik(target, base)
        pos_error, ori_error = get_error(stats)
        target_reached = is_valid(ik_soln) and pos_error < 10e-3 and ori_error < 10e-3
        print(y, target_reached)

def main():
    # test_ik_checker()
    characterize_tablet_workspace()

if __name__ == '__main__':
    main()

import numpy as np
import sophuspy as sp
from scipy.spatial.transform import Rotation as R

from kinematics import TabletPlanner, generate_test_human

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize
from plot_tools import plot_coordinate_frame, plot_coordinate_frame_2d, plot_base_reachability

import json

PI_2 = np.pi / 2.

def is_valid(q):
    if q["lift"] < 0.1 or q["lift"] > 1.2:
        return False
    if q["arm_extension"] < 0.01 or q["arm_extension"] > 0.5:
        return False
    # if q["yaw"] < np.deg2rad(-75.) or q["yaw"] > np.pi:
    if q["yaw"] < -1.75 or q["yaw"] > 4.:
        return False
    if q["pitch"] < -PI_2 or q["pitch"] > np.deg2rad(30.):
        return False
    if np.abs(q["roll"]) > np.pi:
        return False

    return True

def get_error(stats_dict: dict):
    error = stats_dict["final_error"]
    pos_error = error[:3]
    ori_error = error[3:]
    pos_error_norm = np.linalg.norm(pos_error)
    ori_error_norm = np.linalg.norm(ori_error)
    return pos_error_norm, ori_error_norm

def characterize_tablet_workspace_multiple(save: bool=False, save_filename="test.json"):
    # init planner
    planner = TabletPlanner()
    human = generate_test_human("/home/lamsey/ament_ws/src/stretch_tablet/data/matt/")
    targets = planner.generate_tablet_view_points()

    # targets to human head frame
    human_head_root = human.pose_estimate.body_estimate["nose"]
    human_head_root_world = human.pose_estimate.get_point_world(human_head_root)
    human_head_pose = sp.SE3([[0,-1,0], [1,0,0], [0,0,1]], human_head_root_world)
    targets = [human_head_pose * target for target in targets]

    # init test points
    n = 5
    test_base_x = np.linspace(-1.5, 1.5, n)
    test_base_y = np.linspace(-2.5, 0.5, n)

    # init containers
    counts = np.zeros([n, n])
    min_yaw = float('inf')
    max_yaw = -float('inf')

    for k, target in enumerate(targets):
        print('target ', k + 1, 'of', len(targets))
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

                if valid:
                    reachable.append([x, y])
                    min_yaw = min([min_yaw, ik_soln["yaw"]])
                    max_yaw = max([max_yaw, ik_soln["yaw"]])
                    counts[i, j] += 1
                else:
                    unreachable.append([x, y])

    if save:
        dump = {
            "base_x": test_base_x.tolist(),
            "base_y": test_base_y.tolist(),
            "counts": counts.tolist()
        }

        with open(save_filename, 'w') as f:
            json.dump(dump, f)

    # init plot
    f = plt.figure()
    a = f.add_subplot(1, 1, 1, projection='3d')

    plot_base_reachability(a, test_base_x, test_base_y, counts, targets)

    # plot human
    human_pose = human.pose_estimate.get_body_world().T
    human_pose = [p for p in human_pose if np.linalg.norm(p[:2]) > 0.1]
    human_pose = np.array(human_pose).T
    
    a.scatter(*human_pose, color='k', s=30, alpha=1, zorder=10)
    a.scatter(*human.pose_estimate.get_face_world(), color='k', alpha=1, zorder=10)
    
    # cfg
    a.set_xlabel('x (m)')
    a.set_ylabel('y (m)')
    a.set_zlabel('z (m)')
    a.view_init(35, 65)
    a.set_aspect('equal')

    plt.show()

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

def characterize_tablet_workspace_cost():
    # init planner
    planner = TabletPlanner()
    human = generate_test_human("/home/lamsey/ament_ws/src/stretch_tablet/data/matt/")
    targets = planner.generate_tablet_view_points()

    # targets to human head frame
    human_head_root = human.pose_estimate.body_estimate["nose"]
    human_head_root_world = human.pose_estimate.get_point_world(human_head_root)
    human_head_pose = sp.SE3([[0,-1,0], [1,0,0], [0,0,1]], human_head_root_world)
    # human_head_pose = sp.SE3(np.eye(3), human_head_root_world)
    targets = [human_head_pose * target for target in targets]

    # init test points
    n = 50
    test_base_x = np.linspace(-1.5, 1.5, n)
    test_base_y = np.linspace(-2.5, 0.5, n)
    plot_x, plot_y = np.meshgrid(test_base_x, test_base_y)

    # init containers
    costs = np.ones([n, n])
    min_cost = float('inf')
    best_soln = {}

    for k, target in enumerate(targets):
        print('target ', k + 1, 'of', len(targets))

        for i in range(n):
            x = test_base_x[i]
            for j in range(n):
                y = test_base_y[j]
                base = sp.SE3(np.eye(3), np.array([x, y, 0]))
                ik_soln, stats = planner.ik(target, base)
                # print(ik_soln)
                
                pos_error, ori_error = get_error(stats)
                valid = is_valid(ik_soln) and pos_error < 10e-3 and ori_error < 10e-3
                if valid:
                    cost = planner.cost_midpoint_displacement(ik_soln)
                    costs[i, j] = cost
                    if cost < min_cost:
                        min_cost = cost
                        best_soln = ik_soln
                    
                    ik_soln, stats = planner.ik(target, base, debug=True)
                    input()

                else:
                    costs[i, j] = float('inf')
        
        # get lowest cost
        # costs[costs.isna] = float('inf')
        min_cost = np.min(costs)
        # print(min_cost)
        min_cost_ij = np.unravel_index(np.argmin(costs, axis=None), costs.shape)
        # print(min_cost_ij)
        min_cost_x = plot_x[min_cost_ij]
        min_cost_y = plot_y[min_cost_ij]
        print(min_cost_x, min_cost_y, ":", min_cost)
        print(json.dumps(best_soln, indent=2))
        # return

        f = plt.figure()
        a = f.add_subplot()
        a.set_title(str(target.translation()))
        a.scatter(plot_x, plot_y, c=costs)
        a.scatter(min_cost_x, min_cost_y, s=60, c='r', marker='x')
        plot_coordinate_frame_2d(a, target.translation(), target.rotationMatrix(), l=0.2)

        # human
        human_pose = human.pose_estimate.get_body_world().T
        human_pose = [p for p in human_pose if np.linalg.norm(p[:2]) > 0.1]
        human_pose = np.array(human_pose).T
        
        a.scatter(*human_pose[:2, :], color='k')

        norm = Normalize(vmin=0, vmax=np.max(costs))
        sm = cm.ScalarMappable(cmap=cm.get_cmap('jet'), norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=a)

        a.set_xlim(min(test_base_x), max(test_base_x))
        a.set_ylim(min(test_base_y), max(test_base_y))
        a.set_xlabel('x (m)')
        a.set_ylabel('y (m)')
        a.set_aspect('equal')
        a.set_axisbelow(True)
        plt.grid()

        # f.savefig(str(k+1)+".png")
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
    # characterize_tablet_workspace()
    # characterize_tablet_workspace_multiple()
    characterize_tablet_workspace_cost()

if __name__ == '__main__':
    main()

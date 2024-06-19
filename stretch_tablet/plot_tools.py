import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize

import os
import json

# helpers
def plot_coordinate_frame(a, p, R, l):
    x0 = y0 = z0 = p
    x1 = x0 + R[:, 0] * l
    y1 = y0 + R[:, 1] * l
    z1 = z0 + R[:, 2] * l

    x = np.vstack([x0, x1]).T
    y = np.vstack([y0, y1]).T
    z = np.vstack([z0, z1]).T

    a.plot(*x, color='r')
    a.plot(*y, color='g')
    a.plot(*z, color='b')

def points2np(points: list):
    points_np = np.zeros([3, len(points)])
    for i in range(len(points)):
        point = points[i]
        points_np[:, i] = [point["x"], point["y"], point["z"]]

    return points_np

def plot_base_reachability(a, base_x, base_y, counts, targets):
    a.computed_zorder = False

    plot_x, plot_y = np.meshgrid(base_x, base_y)
    plot_z = np.zeros_like(plot_x)

    # set up custom colormap
    cmap_brg = cm.get_cmap('brg')
    custom_brg = cmap_brg(np.linspace(0.5, 1, len(targets) + 1))
    custom_brg = ListedColormap(custom_brg, N=len(targets))

    # plot floor
    a.scatter(plot_x, plot_y, plot_z, c=counts, cmap=custom_brg, edgecolors='none', alpha=0.75, zorder=-1)

    # colorbar
    norm = Normalize(vmin=0, vmax=len(targets))
    sm = cm.ScalarMappable(cmap=custom_brg, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=a)

    # targets
    for target in targets:
        plot_coordinate_frame(a, target.translation(), target.rotationMatrix(), l=0.1)

# test
def test_plot_coord_frame():
    p = np.array([1, 1, 1]).T
    R = np.eye(3)
    l = 0.1

    f = plt.figure()
    ax = f.add_subplot(projection='3d')
    
    plot_coordinate_frame(ax, p, R, l)

    plt.show()

# main
def main(data_dir: str):
    file = os.path.join(data_dir, "body_1.json")

    with open(file) as f:
        test_data = json.load(f)

    print(json.dumps(test_data, indent=2))
    # return
    for face in test_data:
        points = face["points"]
        points_np = points2np(points)
        print(face["pose"], points_np.shape)
        # print(points_np)

    f = plt.figure()
    ax = f.add_subplot(projection='3d')
    ax.scatter(*points_np)
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)

    args = parser.parse_args()
    
    main(args.data_dir)
    # test_plot_coord_frame()
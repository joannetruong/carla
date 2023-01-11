import numpy as np

import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict
from queue import Queue
from functools import lru_cache

from visenv.sim_world import CarlaSimWorld

from queue import PriorityQueue

MAX_PATH_LEN = 25.0


def a_search(start, goal):
    visited = set()
    queue = PriorityQueue()
    queue.put((np.linalg.norm(xys[start] - xys[goal]), start, [start]))

    while queue.qsize():
        cost, node, path = queue.get()
        cost -= np.linalg.norm(xys[node] - xys[goal])
        if cost > MAX_PATH_LEN:
            return -1
        if node not in visited:
            visited.add(node)

            if node == goal:
                return path
            for child in get_nearest(node):
                if child not in visited:
                    node_xy = xys[node]
                    child_xy = xys[child]
                    heuristic = np.linalg.norm(child_xy - xys[goal])
                    total_cost = cost + np.linalg.norm(node_xy - child_xy) + heuristic
                    queue.put((total_cost, child, path + [child]))
    return -1


MAX_STEP_RANGE = 3.0


@lru_cache(maxsize=None)
def get_nearest(point_i):
    d = np.linalg.norm(xys - xys[point_i, :], axis=1)
    less = d < MAX_STEP_RANGE
    neighbor_points = list(np.where(less)[0])
    return neighbor_points


xys = None


def main():
    world_loc = '/Game/Carla/Maps/'
    world_names = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
    for world_name in world_names:
        simulator = CarlaSimWorld(world_loc + world_name)
        ### WARGNING ###
        # WELP, i somehow totally missed the cast_ray in CARLA API
        # instead of stampling at this strange workaround
        # you can simply cast a grid of rays to identify the sidewalk
        # edit: it was actually just added :) in CARLA 0.9.11
        sample_xs, sample_ys, sample_zs = [], [], []
        for _ in range(20000):
            loc = simulator.world.get_random_location_from_navigation()
            sample_xs.append(loc.x)
            sample_ys.append(loc.y)
            sample_zs.append(loc.z)

        min_x = min(sample_xs)
        min_y = min(sample_ys)
        max_x = max(sample_xs)
        max_y = max(sample_ys)

        xy_arr = np.array([sample_xs, sample_ys]).T

        # reduce density
        nx = int((max_x - min_x) / 2.0)
        ny = int((max_y - min_y) / 2.0)
        print(min_x, max_x, nx)
        print(min_y, max_y, ny)
        xs = np.linspace(min_x, max_x, nx)
        ys = np.linspace(min_y, max_y, ny)
        xv, yv = np.meshgrid(xs, ys)

        unique_ixs = set()
        for x_i in range(xv.shape[0]):
            for y_i in range(xv.shape[1]):
                x, y = xv[x_i, y_i], yv[x_i, y_i]
                xy_vec = np.array([x, y])
                d = np.linalg.norm(xy_arr - xy_vec, axis=1)
                ix = np.argmin(d)
                unique_ixs.add(ix)

        print("Reduced from: ", len(sample_xs), " to: ", len(unique_ixs))

        N_points = len(unique_ixs)
        global xys
        xys = np.zeros((N_points, 3))
        for i, ix in enumerate(unique_ixs):
            xys[i, :] = sample_xs[ix], sample_ys[ix], sample_zs[ix]

        # plt.plot(xys[:,0], xys[:,1],'.r')
        # plt.show()

        np.save(f'carla_sidewalk_goals/{world_name}.npy', xys)

        # generate neighbor list (withing range)
        MAX_RANGE = 15.0
        MIN_RANGE = 10.0
        neighbors = []
        for point_i in tqdm.trange(xys.shape[0]):
            d = np.linalg.norm(xys - xys[point_i, :], axis=1)
            less = d < MAX_RANGE
            greater = d > MIN_RANGE
            union = np.bitwise_and(less, greater)
            neighbor_points = list(np.where(union)[0])

            reachable_neighbors = []
            for goal_i in neighbor_points:
                path = a_search(point_i, goal_i)
                if path != -1: reachable_neighbors.append(goal_i)

            print(len(neighbor_points), len(reachable_neighbors))

            neighbors.append(reachable_neighbors)
        assert len(neighbors) == xys.shape[0]
        if 0:
            ix = 2000
            x, y, z = xys[ix]
            plt.plot(x, y, 'ob')
            for neighbor in neighbors[ix]:
                x, y, z = xys[neighbor]
                plt.plot(x, y, '.r')
            plt.show()
        pickle.dump(neighbors, open(f"carla_sidewalk_goals/{world_name}.pickle", "wb"))


if __name__ == '__main__':
    main()
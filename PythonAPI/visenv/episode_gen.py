import numpy as np

import tqdm

import json

from functools import lru_cache

from sim_world import CarlaSimWorld

from queue import PriorityQueue

MAX_PATH_LEN = 25.0
def a_search(start, goal):
    visited = set()
    queue = PriorityQueue()
    queue.put((np.linalg.norm(xys[start]-xys[goal]), start, [start]))

    while queue.qsize():
        cost, node, path = queue.get()
        cost -= np.linalg.norm(xys[node]-xys[goal])
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
                    heuristic = np.linalg.norm(child_xy-xys[goal])
                    total_cost = cost + np.linalg.norm(node_xy-child_xy) + heuristic
                    queue.put((total_cost, child, path + [child]))
    return -1

MAX_STEP_RANGE=3.0
@lru_cache(maxsize=None)
def get_nearest(point_i):
    d = np.linalg.norm(xys-xys[point_i,:],axis=1)
    less = d < MAX_STEP_RANGE
    neighbor_points = list(np.where(less)[0])
    return neighbor_points

xys = None

def main():
    world_loc = '/Game/Carla/Maps/'
    world_names = ["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]
    for world_name in world_names:
        simulator = CarlaSimWorld(world_loc + world_name)
        min_dist = 20
        max_dist = 150
        eps_per_scene = 100

        scene_dataset = {}
        scene_dataset["episodes"] = []
        valid_ctr = 1
        for ctr in range(eps_per_scene*100):
            if valid_ctr > eps_per_scene:
                break
            start = simulator.world.get_random_location_from_navigation()
            goal = simulator.world.get_random_location_from_navigation()
            start_np = np.array([start.x, start.y, start.z])
            goal_np = np.array([goal.x, goal.y, goal.z])
            dist = np.linalg.norm(start_np - goal_np)
            if (dist > max_dist) or (dist < min_dist):
                pass
            else:
                episode = {}
                episode["episode_id"] = valid_ctr
                episode["world_name"] = world_name
                episode["start_position"] = start_np.tolist()
                episode["goal_position"] = goal_np.tolist()
                episode["euclid_dist"] = dist
                scene_dataset["episodes"].append(episode)
                valid_ctr +=1


        print('num episodes: ', len(scene_dataset["episodes"]))
        with open(f'carla_sidewalk_goals/{world_name}.json', "w") as outfile:
            json.dump(scene_dataset, outfile)

if __name__ == '__main__':
    main()
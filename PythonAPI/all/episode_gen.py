import numpy as np
import random
import os
import tqdm

import json

from functools import lru_cache

from sim_world import CarlaSimWorld
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

from queue import PriorityQueue

import carla


def main():
    world_loc = "/Game/Carla/Maps/"
    world_names = [
        "Town01",
        "Town02",
        "Town03",
        "Town04",
        "Town05",
        "Town06",
        "Town07",
        "Town10HD",
    ]

    os.makedirs("carla_sidewalk_goals", exist_ok=True)
    for world_name in world_names:
        simulator = CarlaSimWorld(world_loc + world_name)
        dao = GlobalRoutePlannerDAO(simulator.world.get_map(), 2)
        grp = GlobalRoutePlanner(dao)
        grp.setup()

        min_dist = 1
        max_dist = 150
        eps_per_scene = 100

        scene_dataset = {}
        scene_dataset["episodes"] = []
        valid_ctr = 1
        geo_dists = []
        done = False
        while not done:
            if valid_ctr > eps_per_scene:
                done = True
            start = random.choice(simulator.world.get_map().get_spawn_points())
            goal = random.choice(simulator.world.get_map().get_spawn_points())
            # start = simulator.world.get_random_location_from_navigation()
            # goal = simulator.world.get_random_location_from_navigation()
            start_np = np.array([start.location.x, start.location.y, start.location.z])
            goal_np = np.array([goal.location.x, goal.location.y, goal.location.z])

            route = grp.trace_route(start.location, goal.location)
            waypoint_list = []
            for waypoint in range(len(route)):
                wpt = route[waypoint][0]
                waypoint_list.append(
                    np.array(
                        [
                            wpt.transform.location.x,
                            wpt.transform.location.y,
                            wpt.transform.location.z,
                        ]
                    )
                )

            geo_dist = 0
            for wpt_s, wpt_g in zip(waypoint_list[:-1], waypoint_list[1:]):
                distance = np.linalg.norm(wpt_s - wpt_g)
                geo_dist += distance

            # dist = np.linalg.norm(start_np - goal_np)
            if (geo_dist > max_dist) or (geo_dist < min_dist):
                pass
            else:
                episode = {}
                episode["episode_id"] = valid_ctr
                episode["world_name"] = world_name
                episode["start_position"] = start_np.tolist()
                episode["goal_position"] = goal_np.tolist()
                episode["geodesic_dist"] = geo_dist
                scene_dataset["episodes"].append(episode)
                valid_ctr += 1
                print(geo_dist)
                geo_dists.append(geo_dist)
        print(
            f"num episodes: {len(scene_dataset['episodes'])} "
            f"avg geo dist: {np.mean(np.array(geo_dists))}"
        )
        with open(f"carla_sidewalk_goals/{world_name}.json", "w") as outfile:
            json.dump(scene_dataset, outfile)


if __name__ == "__main__":
    main()

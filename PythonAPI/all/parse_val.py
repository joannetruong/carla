import json
import argparse
import os
import numpy as np
import glob


def main():
    argparser = argparse.ArgumentParser(description="CARLA Automatic Control Client")
    argparser.add_argument("eval_dir")
    args = argparser.parse_args()
    successes = []
    spls = []
    num_actions = []
    num_collisions = []
    dist2goals = []
    episode_dists = []
    geodesic_dists = []
    for town_json in glob.glob(os.path.join(args.eval_dir, "*.json")):
        f = open(town_json)
        results_data = json.load(f)

        for result in results_data["results"]:
            successes.append(result["success"] * 100)
            spls.append(result["spl"] * 100)
            num_actions.append(result["num_actions"])
            num_collisions.append(result["num_collisions"])
            dist2goals.append(result["dist2goal"])
            episode_dists.append(result["episode_dist"])
            geodesic_dists.append(result["geodesic_dist"])

        # assert len(successes) == 12
        # assert len(spls) == 12
        # assert len(num_actions) == 12
        # assert len(num_collisions) == 12
        # assert len(dist2goals) == 12
        # assert len(episode_dists) == 12
        # assert len(geo_dists) == 12

    print(f"Total # episodes: {len(successes)}")
    print(
        f"Avg SR: {np.mean(successes):0.2f} \n"
        f"Avg SPL: {np.mean(spls):0.2f} \n"
        f"Avg # ACTIONS: {np.mean(num_actions):0.2f} \n"
        f"Avg # COLLISIONS: {np.mean(num_collisions):0.2f} \n"
        f"Avg DIST2GOAL: {np.mean(dist2goals):0.2f} \n"
        f"Avg EPISODE_DIST: {np.mean(episode_dists):0.2f} \n"
        f"Avg GEO_DIST: {np.mean(geodesic_dists):0.2f} \n"
    )


if __name__ == "__main__":
    main()

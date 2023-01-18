from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import time
import weakref

import cv2
from scipy import ndimage
from real_policy import ContextNavPolicy
from abstract_agent import AbstractAgent

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

try:
    import numpy as np
except ImportError:
    raise RuntimeError("cannot import numpy, make sure numpy package is installed")

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/carla"
    )
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from automatic_control import *
import gym
from queue import Queue

from sim_world import CarlaSimWorld
import json

IM_W = 128
IM_H = 256
SENSOR_HEIGHT = 0.65
FOV = 58.286
MAX_DEPTH = 3.5
MIN_DEPTH = 0.0
MAP_RES = 100
from gym import spaces
import torch

# from real_policy import NavPolicy, ContextNavPolicy
# from agents.navigation.pointnav_agent import PointNavAgent
from agents.navigation.behavior_agent import (
    BehaviorAgent,
)  # pylint: disable=import-error
from agents.navigation.roaming_agent import RoamingAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error

#


class EvalWorld(gym.Env):
    def __init__(self, world_name, server_port=2000):
        self.client = carla.Client("localhost", server_port)
        self.client.set_timeout(2.0)

        self.world = self.client.get_world()
        self.world = self.client.load_world(world_name)
        self.blueprint_library = self.world.get_blueprint_library()
        self.actor_list = []
        self.world_name = world_name
        self.depth_camera = None
        self.right_depth_queue = Queue()
        self.left_depth_queue = Queue()
        self.collision_queue = Queue()
        self.map_queue = Queue()
        self.height = 0.1
        self.init_weather()
        # self.init_settings()

        pygame.init()
        pygame.font.init()

        self.camera_manager = None
        self.hud = HUD(1280, 720)
        self._gamma = 2.2

    def init_settings(self):
        settings = self.world.get_settings()
        # settings.fixed_delta_seconds = 0.5
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

    def init_weather(self):
        weather = carla.WeatherParameters.ClearNoon
        self.world.set_weather(weather)

    def init_robot(self, start_position):
        bp = random.choice(self.blueprint_library.filter("vehicle.mercedes-benz.coupe"))
        #
        # # A blueprint contains the list of attributes that define a vehicle's
        # # instance, we can read them and modify some of them. For instance,
        # # let's randomize its color.
        # if bp.has_attribute('color'):
        #     color = random.choice(bp.get_attribute('color').recommended_values)
        #     bp.set_attribute('color', color)

        # bp = self.blueprint_library.filter('static.prop.constructioncone')[0]
        # bp = self.blueprint_library.filter('static.prop.constructioncone')[0]
        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        # transform = random.choice(self.world.get_map().get_spawn_points())
        transform = carla.Transform(
            # carla.Location(x=-0.03740343144695029, y=-0.4164822634134684, z=0.5),
            carla.Location(x=start_position[0], y=start_position[1], z=0.3),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        # So let's tell the world to spawn the vehicle.
        self.robot = self.world.spawn_actor(bp, transform)
        print("spawned robot ", self.robot)

        self.camera_manager = CameraManager(self.robot, self.hud, self._gamma)
        self.camera_manager.transform_index = 0
        self.camera_manager.set_sensor(0, notify=False)

    def init_camera(self):
        camera_bp = self.blueprint_library.find("sensor.camera.depth")
        right_camera_transform = carla.Transform(
            # carla.Location(x=0.03740343144695029, y=-0.4164822634134684, z=0.5),
            carla.Location(x=0.5, y=0.0, z=0.5),
            carla.Rotation(
                pitch=-25.29999995676659, yaw=33.0299998892056, roll=-15.499998048556108
            ),
        )
        left_camera_transform = carla.Transform(
            # carla.Location(x=-0.03740343144695029, y=-0.4164822634134684, z=0.5),
            carla.Location(x=0.5, y=0.0, z=0.5),
            carla.Rotation(
                pitch=-25.29999995676659, yaw=-33.0299998892056, roll=15.499998048556108
            ),
        )

        camera_bp.set_attribute("image_size_x", str(IM_W))
        camera_bp.set_attribute("image_size_y", str(IM_H))
        camera_bp.set_attribute("fov", str(FOV))
        self.right_depth_camera = self.world.spawn_actor(
            camera_bp, right_camera_transform, attach_to=self.robot
        )
        self.left_depth_camera = self.world.spawn_actor(
            camera_bp, left_camera_transform, attach_to=self.robot
        )

        # self.right_depth_camera.listen(lambda image: image.save_to_disk('_out/right_%06d.png' % image.frame, cc))
        # self.left_depth_camera.listen(lambda image: image.save_to_disk('_out/left_%06d.png' % image.frame, cc))
        self.right_depth_camera.listen(self.right_depth_queue.put)
        self.left_depth_camera.listen(self.left_depth_queue.put)

    def init_topdownmap(self):
        bird_relative_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=50.0),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )
        camera_bp = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        camera_bp.set_attribute("image_size_x", str(MAP_RES))
        camera_bp.set_attribute("image_size_y", str(MAP_RES))
        # camera_bp.set_attribute('orthographic', 'true')
        # camera_bp.set_attribute('OrthoWidth', '1800')
        camera_bp.set_attribute("sensor_tick", str(0.0))
        self.camera = self.world.spawn_actor(
            camera_bp, bird_relative_transform, attach_to=self.robot
        )
        self.camera.listen(self.map_queue.put)

    def init_collision_sensor(self):
        relative_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=0.5),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        obs_bp = self.blueprint_library.find("sensor.other.obstacle")
        obs_bp.set_attribute("distance", str(0.15))
        obs_bp.set_attribute("hit_radius", str(0.2))
        self.obs_sensor = self.world.spawn_actor(
            obs_bp, relative_transform, attach_to=self.robot
        )
        self.collisions = Queue()
        self.obs_sensor.listen(self.collision_queue.put)

    def sample_start_goal_locations(self, ep_id):
        episode = self.episodes[ep_id]
        start_np = episode["start_position"]
        start_carla = carla.Location(x=start_np[0], y=start_np[1], z=start_np[2])
        goal_np = episode["goal_position"]
        goal_carla = carla.Location(x=goal_np[0], y=goal_np[1], z=goal_np[2])
        return start_carla, goal_carla

    def get_sidewalk_height(self, location):
        x, y, z = location
        p = self.world.ground_projection(carla.Location(x, -y, z + self.height * 2))
        if p is not None and p.label == carla.CityObjectLabel.Sidewalks:
            return p.location.z + self.height
        return z

    def get_depth_image(self):
        right_raw = self.right_depth_queue.get()
        left_raw = self.left_depth_queue.get()

        right_img = np.frombuffer(right_raw.raw_data, dtype=np.dtype("uint8")).astype(
            np.float32
        )
        right_img = np.reshape(right_img, (IM_H, IM_W, -1))[:, :, :3]

        left_img = np.frombuffer(left_raw.raw_data, dtype=np.dtype("uint8")).astype(
            np.float32
        )
        left_img = np.reshape(left_img, (IM_H, IM_W, -1))[:, :, :3]

        depth_img = np.concatenate((left_img, right_img), axis=1)

        normalized = (
            depth_img[:, :, 2]
            + depth_img[:, :, 1] * 256
            + depth_img[:, :, 0] * 256 * 256
        ) / (256 * 256 * 256 - 1)
        depth_img_meters = 1000 * normalized
        depth_img_meters[depth_img_meters > MAX_DEPTH] = 255.0
        depth_img_meters[depth_img_meters == 0.0] = 255.0
        depth_img_meters = np.clip(depth_img_meters, MIN_DEPTH, MAX_DEPTH)
        depth_img_meters = depth_img_meters / MAX_DEPTH
        depth_img_meters *= 255.0

        return depth_img_meters

    def get_occupancy_map(self):
        img = self.map_queue.get()
        orig_grid = np.reshape(img.raw_data, (MAP_RES, MAP_RES, 4))[:, :, 2]
        grid = np.copy(orig_grid)

        # 1 - Building
        # 2 - Fence
        # 3 - Other
        # 5 - Pole
        # 7 - Road
        # 11 - Wall
        # 12 - TrafficSign
        # 18 - TrafficLight
        # 19 - TrafficLight
        # 20 - Dynamic

        for sem_id in [1, 2, 3, 5, 7, 11, 12, 18, 19, 20]:
            dilated_mask = ndimage.binary_dilation(grid == sem_id, iterations=1)
            grid[dilated_mask] = sem_id

        # 4 - Pedestrian
        # 8 - Sidewalk [x]
        # 9 - Vegetation - usually just causes occlusion
        # 22 - Terrain ---  is ground level, and non-walkable
        # AVOID for now, makes the away from sidewalk a sidewalk
        # for sem_id in [9]:
        #     whole_mask = ndimage.binary_erosion(grid==sem_id,iterations=2)
        #     dilated_region = np.bitwise_xor(whole_mask,grid==sem_id)
        #     grid[dilated_region] = 8 # make sidewalk (naive)

        filt_grid = np.zeros_like(grid)
        # Set pixels to 1 if corresponding value in original grid is 7 or 8
        filt_grid[np.logical_or(grid == 7, grid == 8)] = 1

        mask = np.where(grid == 7)
        grid = (
            grid == 8
        )  # np.bitwise_or(grid == 4,grid == 8) # FIXME currently removes agent
        # grid[grid == 7] = True
        # np.bitwise_or(grid == 7, grid == 8)
        grid[62:66, 63:65] = True  # remove cone
        grid[63:65, 62:66] = True  # remove cone
        grid[62:66, 63:65] = False  # draw robot box

        return orig_grid, filt_grid.astype(np.bool)

    def get_pos_rot(self):
        transform = self.robot.get_transform()
        loc = transform.location
        rot = transform.rotation
        x = loc.x
        y = -loc.y
        z = loc.z

        # rotation in degrees
        roll = np.deg2rad(rot.roll)
        pitch = np.deg2rad(rot.pitch)
        yaw = np.deg2rad(rot.yaw)
        xyz, rpy = np.array([x, y, z]), np.array([roll, pitch, yaw])
        return xyz, rpy

    def set_pos_rot(self, xyz, rpy):
        # SEND COMMAND
        location = carla.Location()
        rotation = carla.Rotation()
        location.x = xyz[0]
        location.y = -xyz[1]
        location.z = xyz[2]

        # Rotation is in degrees
        rotation.roll = np.rad2deg(rpy[0])
        rotation.pitch = np.rad2deg(rpy[1])
        rotation.yaw = np.rad2deg(rpy[2])

        transform = carla.Transform(location, rotation)
        self.robot.set_transform(transform)

    def calculate_new_pos_rot(self, actions, xyz, rpy):
        lin_vel, hor_vel, ang_vel = actions
        curr_yaw = rpy[-1]

        rot_m = np.array(
            [
                [np.cos(curr_yaw), -np.sin(curr_yaw)],
                [np.sin(curr_yaw), np.cos(curr_yaw)],
            ]
        )
        vel = np.array([lin_vel, hor_vel])
        local_vel = np.dot(rot_m, vel)
        xyz[0] += local_vel[0]
        xyz[1] += local_vel[1]
        xyz[2] = self.get_sidewalk_height(xyz)
        rpy[0] = 0
        rpy[1] = 0
        rpy[2] = self.wrap_heading(curr_yaw + ang_vel)
        # print(
        #     "CURR THETA: ",
        #     np.rad2deg(curr_yaw),
        #     "NEW YAW: ",
        #     np.rad2deg(curr_yaw + ang_vel),
        #     np.rad2deg(rpy[2]),
        # )

        # rpy[2] = 0
        return xyz, rpy

    def apply_control(self, actions):
        xyz, rpy = self.get_pos_rot()
        new_xyz, new_rpy = self.calculate_new_pos_rot(actions, xyz, rpy)
        self.set_pos_rot(new_xyz, new_rpy)

    def wrap_heading(self, heading):
        """Ensures input heading is between -180 an 180; can be float or np.ndarray in radians"""
        return (heading + np.pi) % (2 * np.pi) - np.pi

    def _compute_pointgoal(self, goal_xy):
        # Get rho theta observation
        xyz, rpy = self.get_pos_rot()
        self.x = xyz[0]
        self.y = xyz[1]
        self.yaw = rpy[-1]
        curr_xy = np.array([self.x, self.y], dtype=np.float32)
        rho = np.linalg.norm(curr_xy - goal_xy)
        theta = np.arctan2(goal_xy[1] - self.y, goal_xy[0] - self.x) - self.yaw
        theta = np.arctan2(goal_xy[1] - self.y, goal_xy[0] - self.x) - self.yaw
        rho_theta = np.array([np.log(rho), self.wrap_heading(theta)], dtype=np.float32)

        return rho_theta

    def eval_episode(self, display, episode, policy):
        start_position = episode["start_position"]
        goal_position = episode["goal_position"]
        episode_geo_dist = episode["geodesic_dist"]

        print(
            f"start_position: {start_position}"
            f"goal_position: {goal_position}"
            f"geodesic_dist: {episode_geo_dist}"
        )

        MAX_STEPS = 1000
        SUCCESS_DISTANCE = 0.425
        DEBUG = False
        num_actions = 0
        num_collisions = 0
        done = False
        success = False
        episode_distance = 0

        while not done:
            depth_img = self.get_depth_image()
            orig_grid, context_map = self.get_occupancy_map()

            if DEBUG:
                cv2.imwrite(f"_out/depth_{num_actions}.png", depth_img)
                cv2.imwrite(
                    f"_out/map_{num_actions}.png", context_map.astype(np.uint8) * 255
                )

            collision = False
            curr_xyz, curr_rpy = self.get_pos_rot()

            if not self.collision_queue.empty():
                collision = self.collision_queue.get()
            if collision:
                num_collisions += 1
                print(f"COLLISION! {num_collisions}")
                self.set_pos_rot(curr_xyz, curr_rpy)
            else:
                observations = {}
                gps_compass = self._compute_pointgoal(goal_position[:2])
                split_depth_img = np.split(depth_img, 2, axis=1)
                observations["spot_left_depth"] = np.expand_dims(
                    split_depth_img[1], 2
                ).astype("float32")
                observations["spot_right_depth"] = np.expand_dims(
                    split_depth_img[0], 2
                ).astype("float32")
                observations["pointgoal_with_gps_compass"] = gps_compass
                observations["context_map"] = context_map

                actions = policy.act(observations, deterministic=True)
                print("actions: ", actions)
                # # lin_vel, hor_vel, ang_vel (in radians)
                # lin_vel = 1.0
                # ang_vel = np.deg2rad(0.0)
                # actions = np.array([lin_vel, 0.0, ang_vel])
                self.apply_control(actions)

            self.world.tick()
            # world.world.wait_for_tick(10.0)
            new_xyz, new_rpy = self.get_pos_rot()
            # print("curr_xyz:", curr_xyz, "new_xyz: ", new_xyz)
            num_actions += 1
            dist_to_goal = np.linalg.norm(new_xyz[:2] - goal_position[:2])
            episode_distance += np.linalg.norm(new_xyz[:2] - curr_xyz[:2])
            if dist_to_goal < SUCCESS_DISTANCE:
                done = True
                success = True
            if num_actions > MAX_STEPS:
                done = True
                success = False

            # print('moved vehicle to:', np.linalg.norm(new_xyz - curr_xyz), np.rad2deg(np.abs(curr_rpy[-1] - new_rpy[-1])))
            # world.tick(clock)
            # world.render(display)
            self.camera_manager.render(display)
            self.hud.render(display)

            pygame.display.flip()
        print(
            f"EPISODE TERMINATED. "
            f"SUCCESS: {success} "
            f"SPL: {success * (episode_distance / np.max(episode_distance, episode_geo_dist))} "
            f"# ACTIONS: {num_actions}, "
            f"DIST2GOAL: {dist_to_goal} "
        )

    def init_world(self, args, episode):
        start_position = episode["start_position"]

        display = pygame.display.set_mode(
            (args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        # self.hud = HUD(args.width, args.height)
        # world = World(self.client.get_world(), self.hud, args)
        self.init_robot(start_position)
        self.init_camera()
        self.init_topdownmap()
        self.init_collision_sensor()

        clock = pygame.time.Clock()
        return display

    def destroy_actors(self):
        print("destroying actors")
        self.right_depth_camera.destroy()
        self.left_depth_camera.destroy()
        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in self.actor_list]
        )
        print("done.")

    def game_loop(self, args, episode, weights):
        ## load policy
        policy = eval(args.policy_type)(weights)
        policy.reset()

        ## run eval
        try:
            display = self.init_world(args, episode)
            self.eval_episode(display, episode, policy)
            time.sleep(5)
        finally:
            self.destroy_actors()


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(description="CARLA Automatic Control Client")
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="debug",
        help="Print debug information",
    )
    argparser.add_argument(
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    argparser.add_argument(
        "--res",
        metavar="WIDTHxHEIGHT",
        default="1280x720",
        help="Window resolution (default: 1280x720)",
    )
    argparser.add_argument(
        "-pt",
        "--policy-type",
        type=str,
        choices=["ContextNavPolicy", "NavPolicy"],
        help="select which agent to run",
        default="ContextNavPolicy",
    )
    argparser.add_argument(
        "-s",
        "--seed",
        help="Set seed for repeating executions (default: None)",
        default=None,
        type=int,
    )

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split("x")]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    logging.info("listening to server %s:%s", args.host, args.port)

    print(__doc__)

    world_loc = "/Game/Carla/Maps/"
    episodes_loc = "/home/joanne/repos/carla/PythonAPI/all/carla_sidewalk_goals"
    world_names = ["Town01"]
    # world_names = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]

    weights_dir = "/home/joanne/repos/carla/PythonAPI/all/weights/"
    policy_pth = "spot_depth_context_resnet18_map_prevact_sincos32_log_rot_100_-1.0_cma_bp_0.3_sd_1_ckpt_93.pth"
    weights = os.path.join(weights_dir, policy_pth)

    for world_name in world_names:
        simulator = EvalWorld(os.path.join(world_loc, world_name))
        episode_json = os.path.join(episodes_loc, world_name + ".json")
        f = open(episode_json)
        episodes_data = json.load(f)
        for episode in episodes_data["episodes"]:
            try:
                simulator.game_loop(args, episode, weights)
            except KeyboardInterrupt:
                print("\nCancelled by user. Bye!")


if __name__ == "__main__":
    main()

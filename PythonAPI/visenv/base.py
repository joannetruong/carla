import glob
import os
import sys
import copy
import random
import pickle
from queue import Queue
from collections import OrderedDict

import gym
from gym import spaces
from matplotlib.pyplot import imshow
import numpy as np
import cv2
from scipy import ndimage

# from sim_world import CARLA_DIR
from obstacles_list import obstacle_bp_names
# from common.utils import ranged_angle, load_yaml
# from common.img_lidar_scan import img_lidar_scan

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from sim_world import CarlaSimWorld
from agents import *

import torch
# import third_party.BiSeNet.lib.transform_cv2 as T
# from third_party.BiSeNet.lib.models import model_factory
# from third_party.BiSeNet.configs import cfg_factory

# TODO move to config
IM_H = 128
IM_W = IM_H

# LIDAR frame remove regions
FRAME1 = [35,57]
FRAME2 = [125,147]
FRAME3 = [215,237]
FRAME4 = [305,327]
FRAME = [fi for F in [FRAME1,FRAME2,FRAME3,FRAME4] for fi in range(F[0],F[1])]

train_weather = [
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.CloudyNoon,
    carla.WeatherParameters.WetNoon,
    carla.WeatherParameters.WetCloudyNoon,
    carla.WeatherParameters.SoftRainNoon,
    # carla.WeatherParameters.MidRainyNoon,
    # carla.WeatherParameters.HardRainNoon,
    # carla.WeatherParameters.ClearSunset,
    # carla.WeatherParameters.CloudySunset,
    # carla.WeatherParameters.WetSunset,
    carla.WeatherParameters.WetCloudySunset,
    carla.WeatherParameters.SoftRainSunset,
    carla.WeatherParameters.MidRainSunset,
    carla.WeatherParameters.HardRainSunset
]

test_weather = [
    carla.WeatherParameters.MidRainyNoon,
    carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.ClearSunset,
    carla.WeatherParameters.CloudySunset,
    carla.WeatherParameters.WetSunset,
]

class VisualWorldEnv(gym.Env):
    def __init__(self, world_name, server_port, config, goal_visible=False,render_info=False):
        super(VisualWorldEnv, self).__init__()
        self.config = config
        self.grid_config = load_yaml(config['grid_env_config'])
        self.student_mods = [*config['student_obs']['flat'].keys(),*config['student_obs']['camera'].keys()]
        self.world_name = world_name
        self.render_info = render_info
        self.goal_visible = goal_visible
        self.simulator = CarlaSimWorld("/Game/Carla/Maps/" + self.world_name, server_port)
        self.spectator = self.simulator.world.get_spectator()
        self.allowed_ids = [ # ids that don't cause sidewalk violation
            carla.CityObjectLabel.NONE,
            carla.CityObjectLabel.Sidewalks,
            carla.CityObjectLabel.Pedestrians,
            ]

        self.init_locations = np.load(f'carla_sidewalk_goals/{self.world_name}.npy').astype(float)
        self.goal_locations = pickle.load(open(f'carla_sidewalk_goals/{self.world_name}.pickle','rb'))

        self.obstacles = []
        for obstacle_i in range(20):
            bp_name = random.choice(obstacle_bp_names)
            bp = self.simulator.BPL.filter(bp_name)[0]
            obstacle = AbstractAgent(
                world=self.simulator.world,
                model_blueprint=bp,
                init_location=carla.Location(0,0,0),
                init_rotation=[0,0,0])
            self.obstacles.append(obstacle)

        self.character = None
        self.goal = None

        self.HEIGHT = IM_H
        self.WIDTH = IM_W
        self.IMG_OBS_HIST = 3

        self.t_sem = Queue()
        self.t_lidar = Queue()
        if 'inf_semantic' in self.student_mods: self.student_mods.append('rgb')
        if 'lidar' in self.student_mods: self.s_lidar = Queue()
        if 'rgb' in self.student_mods: self.s_rgb = Queue()
        if 'depth' in self.student_mods: self.s_dpt = Queue()
        if 'semantic' in self.student_mods: self.s_sem = Queue()

        obs_spaces = {
            'image': spaces.Box(low=0, high=1, shape=(self.IMG_OBS_HIST, self.HEIGHT, self.WIDTH), dtype=np.bool),
            'normed_goal_direction': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'goal_distance': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'grid_lidar': spaces.Box(low=0, high=1, shape=(64,), dtype=np.float32),
            }
        self.observation_space = gym.spaces.Dict(obs_spaces)
        n_actions = 2
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_actions, ), dtype=np.float32)
        
        if 'inf_semantic' in self.student_mods:
            self.to_tensor = T.ToTensor(
                mean=(0.3257, 0.3690, 0.3223), # city, rgb
                std=(0.2112, 0.2148, 0.2115),
            )
            cfg = cfg_factory["bisenetv2"]
            net = model_factory[cfg.model_type](2)
            net.load_state_dict(torch.load(
                "third_party/BiSeNet/sidewalk_vs_bg+carla_sw_480x640_399999.pth",
                map_location='cpu'))
            net.eval()
            net.cuda()
            self.bisenet = net

    def __del__(self):
        if hasattr(self, 'obs_sensor'):
            self.obs_sensor.destroy()
        if hasattr(self, 'camera'):
            self.camera.destroy()
        if hasattr(self, 'camera_rgb'):
            self.camera_rgb.destroy()
        if hasattr(self, 'camera_sem'):
            self.camera_sem.destroy()
        if hasattr(self, 'camera_dpt'):
            self.camera_dpt.destroy()
        if hasattr(self, 'lidar'):
            self.lidar.destroy()
        if hasattr(self, 'real_lidar'):
            self.real_lidar.destroy()

    def init_character(self, loc, rot):
        model_blueprint = self.simulator.BPL.filter("static.prop.constructioncone")[0]
        self.character = AbstractAgent(
            world=self.simulator.world,
            model_blueprint=model_blueprint,
            init_location=loc,
            init_rotation=rot)
        
        # MAKE COLLISION SENSOR
        relative_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=0.4),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            )
        obs_bp = self.simulator.BPL.find('sensor.other.obstacle')
        obs_bp.set_attribute('distance', str(0.15))
        obs_bp.set_attribute('hit_radius', str(0.2))
        self.obs_sensor = self.simulator.world.spawn_actor(obs_bp, relative_transform, attach_to=self.character.model)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        self.collisions=Queue()
        self.obs_sensor.listen(self.collisions.put)

        # ADD SEMANTIC BIRD CAMERA
        bird_relative_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=2.0),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
            )
        camera_bp = self.simulator.BPL.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', str(IM_W))
        camera_bp.set_attribute('image_size_y', str(IM_H))
        camera_bp.set_attribute('orthographic', 'true')
        camera_bp.set_attribute('OrthoWidth', '1800')
        camera_bp.set_attribute('sensor_tick', str(0.0))
        self.camera = self.simulator.world.spawn_actor(camera_bp, bird_relative_transform, attach_to=self.character.model)
        self.camera.listen(self.t_sem.put)

        # ATTACH TEACHER AUG LIDAR SENSOR
        aug_lidar_relative_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=0.4),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            )
        n_channels = 3
        self.aug_n_rays = 360
        self.aug_range = 12.0
        fps = 10
        lidar_bp = self.simulator.BPL.find('sensor.lidar.ray_cast_semantic')
        lidar_bp.set_attribute('channels', str(n_channels))
        lidar_bp.set_attribute('range', str(self.aug_range))
        lidar_bp.set_attribute('points_per_second', str(n_channels*self.aug_n_rays*fps))
        lidar_bp.set_attribute('rotation_frequency', str(fps))
        lidar_bp.set_attribute('upper_fov', str(0.0))
        lidar_bp.set_attribute('lower_fov', str(-10.0))
        lidar_bp.set_attribute('sensor_tick', str(0.0))
        self.lidar = self.simulator.world.spawn_actor(lidar_bp, aug_lidar_relative_transform, attach_to=self.character.model)
        self.lidar.listen(self.t_lidar.put)

        # ATTACH STUDENT AUG LIDAR SENSOR
        real_lidar_relative_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=0.4),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            )
        n_channels = 1
        self.real_n_rays = 360
        self.real_range = 6.0
        fps = 10
        lidar_bp = self.simulator.BPL.find('sensor.lidar.ray_cast_semantic')
        lidar_bp.set_attribute('channels', str(n_channels))
        lidar_bp.set_attribute('range', str(self.real_range))
        lidar_bp.set_attribute('points_per_second', str(n_channels*self.real_n_rays*fps))
        lidar_bp.set_attribute('rotation_frequency', str(fps))
        lidar_bp.set_attribute('upper_fov', str(0.0))
        lidar_bp.set_attribute('lower_fov', str(0.0))
        lidar_bp.set_attribute('sensor_tick', str(0.0))
        self.real_lidar = self.simulator.world.spawn_actor(lidar_bp, real_lidar_relative_transform, attach_to=self.character.model)
        self.real_lidar.listen(self.s_lidar.put)

        relative_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=self.config['sensor_height']),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            )
        if 'rgb' in self.student_mods: # ADD RGB STUDENT CAMERA
            camera_bp = self.simulator.BPL.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.config['img_size']))
            camera_bp.set_attribute('image_size_y', str(self.config['img_size']))
            camera_bp.set_attribute('fov',str(self.config['fov']))
            camera_bp.set_attribute('sensor_tick', str(0.0))
            self.camera_rgb = self.simulator.world.spawn_actor(camera_bp, relative_transform, attach_to=self.character.model)
            self.camera_rgb.listen(self.s_rgb.put)

        if 'depth' in self.student_mods: # ADD DEPTH STUDENT CAMERA
            camera_bp = self.simulator.BPL.find('sensor.camera.depth')
            camera_bp.set_attribute('image_size_x', str(self.config['img_size']))
            camera_bp.set_attribute('image_size_y', str(self.config['img_size']))
            camera_bp.set_attribute('fov',str(self.config['fov']))
            camera_bp.set_attribute('sensor_tick', str(0.0))
            self.camera_dpt = self.simulator.world.spawn_actor(camera_bp, relative_transform, attach_to=self.character.model)
            self.camera_dpt.listen(self.s_dpt.put)

        if 'semantic' in self.student_mods: # ADD SEMANTIC STUDENT CAMERA
            camera_bp = self.simulator.BPL.find('sensor.camera.semantic_segmentation')
            camera_bp.set_attribute('image_size_x', str(self.config['img_size']))
            camera_bp.set_attribute('image_size_y', str(self.config['img_size']))
            camera_bp.set_attribute('fov',str(self.config['fov']))
            camera_bp.set_attribute('sensor_tick', str(0.0))
            self.camera_sem = self.simulator.world.spawn_actor(camera_bp, relative_transform, attach_to=self.character.model)
            self.camera_sem.listen(self.s_sem.put)

    def init_goal(self, loc, rot=[0,0,0]):
        # using another abstract agent to have it via GUI
        # it won't be visible to the main agent
        model_blueprint = self.simulator.BPL.filter("walker.pedestrian.0009")[0]
        model_blueprint = self.simulator.BPL.filter("static.prop.constructioncone")[0]
        #model_blueprint.set_attribute('is_invincible', 'true') # make it stay at the sidewalk level...
        self.goal = GoalAgent(
            world=self.simulator.world,
            model_blueprint=model_blueprint,
            init_location=loc,
            init_rotation=[0,0,0],
            visible=self.goal_visible
        )

    def to_sidewalk_point(self,loc):
        initial_location = carla.Location(*loc)
        initial_location.z += 1
        final_location = carla.Location(*loc)
        final_location.z = -1
        p = self.simulator.world.cast_ray(initial_location, final_location)
        for p_i in p:
            if p_i.label == carla.CityObjectLabel.Sidewalks:
                return p_i.location
        return None

    def sample_start_goal_locations(self,):
        while True:
            start_point_ix = np.random.randint(self.init_locations.shape[0])
            goal_point_ixs = self.goal_locations[start_point_ix]
            if len(goal_point_ixs) != 0:
                start_xyz = self.init_locations[start_point_ix,:]
                goal_point_ix = np.random.choice(goal_point_ixs)
                goal_xyz = self.init_locations[goal_point_ix,:]
                start_xyz = self.to_sidewalk_point(start_xyz)
                if start_xyz is None: continue
                goal_xyz = self.to_sidewalk_point(goal_xyz)
                if goal_xyz is None: continue
                break
        return start_xyz, goal_xyz

    def sim_step_everything(self):
        for _ in range(1):
            self.simulator.step_simulator()
            self.t_sem.get()
            self.t_lidar.get()
            if 'lidar' in self.student_mods: self.s_lidar.get()
            if 'rgb' in self.student_mods: self.s_rgb.get()
            if 'depth' in self.student_mods: self.s_dpt.get()
            if 'semantic' in self.student_mods: self.s_sem.get()
            if not self.collisions.empty(): self.collisions.get()

    def reset(self):
        self.noise_x = 0.0
        self.noise_y = 0.0
        # CHARACTER RESET
        spawned_ok = False
        while not spawned_ok:
            try:
                start_xyz, goal_xyz = self.sample_start_goal_locations()
                rot = [0., 0., np.random.uniform(0, 360)]
                if self.character is None:
                    self.init_character(start_xyz, rot)
                    self.init_goal(goal_xyz)
                start_xyz.z += self.character.height
                self.character.set_via_carla_transform(start_xyz, rot)
                self.goal.set_via_carla_transform(goal_xyz,[0,0,0])
                self.sim_step_everything()
                spawned_ok = self.is_on_sidewalk()
            except Exception as e:
                print('warning: resampling spawn loc')
                print("exception:",e)
        rx,ry = start_xyz.x,start_xyz.y
        size = self.grid_config['obstacles']['sample_area']
        occupied_locs = [[start_xyz.x,start_xyz.y],[goal_xyz.x,goal_xyz.y]]
        for obstacle in self.obstacles:
            attempts = 0
            while True:
                pr = np.random.uniform(low = -np.pi, high = np.pi)
                pd = np.random.uniform(low = 2.0, high = size)
                px = rx + pd * np.cos(pr)
                py = ry + pd * np.sin(pr)
                p = self.simulator.world.ground_projection(carla.Location(px,py,5))
                is_far_from_others = self.loc_is_far_from_locs([px,py],occupied_locs,dist=2.0)
                if p is not None and p.label == carla.CityObjectLabel.Sidewalks and is_far_from_others:
                    new_loc = carla.Location(px,py,p.location.z+0.03)
                    obstacle.set_via_carla_transform(new_loc, [0., 0., np.random.uniform(0, 360)])
                    occupied_locs.append([px,py])
                    break
                attempts+=1
                if attempts>50: break

        self.is_test_town = self.world_name in ["Town03","Town10HD"]
        # random weather
        if self.is_test_town:
            weather_choices = test_weather
        else:
            weather_choices = train_weather
        weather = random.choice(weather_choices)
        self.simulator.world.set_weather(weather)

        # step sim to move pedestrian
        self.sim_step_everything()
        
        self.start_location, self.start_orientation = self.character.get_xyz_rpy()
        self.goal_location, self.goal_orientation = self.goal.get_xyz_rpy()
        
        self.prev_states = None
        #### SENSOR DROPOUT
        self.sensor_dropout = 'no_dropout'
        ####
        observation = self.step(np.zeros_like(self.action_space.sample()))[0]
        return observation  # reward, done, info can't be included

    def loc_is_far_from_locs(self,loc,locs,dist):
        locs = np.array(locs)
        dists = np.linalg.norm(locs-loc,axis=1)
        return np.all(dists > dist)

    def calc_goal_direction(self, state:OrderedDict):
        global_theta = np.arctan2(
            state['goal_location'][1] - state['current_location'][1],
            state['goal_location'][0] - state['current_location'][0])
        local_theta = global_theta - state['current_orientation'][2]
        return ranged_angle(local_theta)

    def get_occupancy_grid(self,):
        img = self.t_sem.get()
        orig_grid = np.reshape(img.raw_data,(IM_H,IM_W,4))[:,:,2]
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
        
        for sem_id in [1,2,3,5,7,11,12,18,19,20]:
            dilated_mask = ndimage.binary_dilation(grid==sem_id,iterations=1)
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
        
        grid = grid == 8#np.bitwise_or(grid == 4,grid == 8) # FIXME currently removes agent
        grid[62:66,63:65] = True # remove cone
        grid[63:65,62:66] = True # remove cone
        grid[62:66,63:65] = False # draw robot box

        return orig_grid, grid.astype(np.bool)

    def get_rgb_image(self,):
        rgb_raw = self.s_rgb.get()
        rgb_raw = np.frombuffer(rgb_raw.raw_data, dtype=np.dtype("uint8"))
        rgb = np.reshape(rgb_raw,(self.config['img_size'],self.config['img_size'],4))[:,:,:3]
        return rgb[120:-40,:,::-1]

    def get_depth_image(self,):
        image = self.s_dpt.get()
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")).astype(np.float32)
        im = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
        normalized = (im[:,:,2] + im[:,:,1] * 256 + im[:,:,0] * 256 * 256) / (256 * 256 * 256 - 1)
        return normalized
    
    def get_inf_semantic_image(self,rgb):
        im = self.to_tensor(dict(im=rgb, lb=None))['im'].unsqueeze(0).cuda()
        out = self.bisenet(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
        out = cv2.resize(out, (256,256), interpolation=cv2.INTER_NEAREST)
        return out==0

    def get_semantic_image(self,):
        sem_raw = self.s_sem.get()
        sem_raw = np.frombuffer(sem_raw.raw_data, dtype=np.dtype("uint8"))
        sem = np.reshape(sem_raw,(self.config['img_size'],self.config['img_size'],4))[:,:,2]
        return sem

    def get_lidar_scan(self,lidar_queue):
        pc_raw = lidar_queue.get()
        data = np.frombuffer(pc_raw.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
        return data

    def is_not_on_sidewalk(self):
        char_tf = self.character.model.get_transform()
        loc = char_tf.location
        theta = char_tf.rotation.yaw * np.pi / 180

        for dx,dy in [(0.3,0.15),(0.3,-0.15),(-0.3,0.15),(-0.3,-0.15)]:
            nx = dx * np.cos(theta) - dy * np.sin(theta)
            ny = dx * np.sin(theta) + dy * np.cos(theta)
            edge_loc = carla.Location(loc.x+nx,loc.y+ny,loc.z+0.1)
            inter = self.simulator.world.ground_projection(edge_loc)
            if (inter is None) or (inter.label not in self.allowed_ids):
                return True
        return False
    
    def is_on_sidewalk(self):
        return not self.is_not_on_sidewalk()
    
    def draw_circle_test(self, grid,x,y,size,color):
        x_min = int(np.clip(x-size//2,0,grid.shape[0]))
        x_max = int(np.clip(x+size//2,0,grid.shape[0]))
        y_min = int(np.clip(y-size//2,0,grid.shape[1]))
        y_max = int(np.clip(y+size//2,0,grid.shape[1]))
        grid[x_min:x_max,y_min:y_max] = color
        return grid
    
    def augment_occupancy_grid(self,grid,pcl_data):
        im_s = grid.shape[0]
        
        occupancy_grid_dim = 18.0
        pcl_xs = -(pcl_data['x']/occupancy_grid_dim*im_s)
        pcl_ys = (pcl_data['y']/occupancy_grid_dim*im_s)

        for cx, cy, sem_id in zip(pcl_xs, pcl_ys, pcl_data['ObjTag']):
            if sem_id == carla.CityObjectLabel.Sidewalks: continue
            grid = self.draw_circle_test(grid, im_s//2+cx, im_s//2+cy, 4, False)
        return grid

    def is_colliding(self):
        is_colliding = False
        if not self.collisions.empty():
            while not self.collisions.empty():
                c = self.collisions.get()
                # c.other is non parent colliding prop (carla.Actor)
                # c.other_actor.semantic_tags
                # c.other_actor.type_id
                # c.distance
                is_colliding = True
        return is_colliding
    
    def pcd_to_bins(self,pcd):
        bins = np.ones((self.real_n_rays))
        pcd_x = pcd['x'] + np.random.uniform(-0.05, 0.05, size=pcd['x'].shape)
        pcd_y = pcd['y'] + np.random.uniform(-0.05, 0.05, size=pcd['y'].shape)
        if pcd.shape[0] != 0:
            rs = np.arctan2(pcd_y,pcd_x)
            rs_normed = (rs / (2*np.pi)) + 0.5
            bins_ixs = np.floor(rs_normed * (self.real_n_rays-1)).astype(int)
            ds = np.sqrt(pcd_x**2+pcd_y**2)
            bins[bins_ixs] = ds / self.real_range
        # FILTER OUT REAL WORLD FRAME OCCLUDED REGION 
        bins = np.delete(bins,FRAME)
        return bins

    def get_state(self,)->OrderedDict:
        state = OrderedDict()
        _, state['occupancy_grid'] = self.get_occupancy_grid()
        aug_point_cloud_data = self.get_lidar_scan(self.t_lidar)
        state['occupancy_grid'] = self.augment_occupancy_grid(state['occupancy_grid'],aug_point_cloud_data)
        state['grid_lidar'] = img_lidar_scan(state['occupancy_grid'])
        if 'lidar' in self.student_mods:
            state['real_lidar_pcd'] = self.get_lidar_scan(self.s_lidar)
            state['real_lidar_polar'] = self.pcd_to_bins(state['real_lidar_pcd'])
        if 'rgb' in self.student_mods: state['rgb_image'] = self.get_rgb_image()
        if 'depth' in self.student_mods: state['depth_image'] = self.get_depth_image()
        if 'semantic' in self.student_mods: state['semantic_image'] = self.get_semantic_image()
        if 'inf_semantic' in self.student_mods:
            state['inf_semantic'] = self.get_inf_semantic_image(state['rgb_image'])
        
        state['sensor_dropout'] = self.sensor_dropout
        state['is_test_town'] = self.is_test_town

        state['start_location'] = self.start_location
        state['current_location'], state['current_orientation'] = self.character.get_xyz_rpy()
        state['goal_location'] = copy.copy(self.goal_location)
        if np.random.rand() < self.config['gps']['noise_update_freq']:
            var = self.config['gps']['std']
            self.noise_x = np.random.normal(0.0,var)
            self.noise_y = np.random.normal(0.0,var)
        state['goal_direction'] = self.calc_goal_direction(state)
        state['global_goal_distance'] = np.linalg.norm(self.goal_location[:2] - state['current_location'][:2])
        state['distance_from_start'] = np.linalg.norm(state['start_location'][:2] - state['current_location'][:2])
        state['is_not_on_sidewalk'] = self.is_not_on_sidewalk()
        state['robot_is_colliding'] = self.is_colliding()
        return state

    def get_observations(self,state:OrderedDict)->OrderedDict:
        obs = OrderedDict()

        occupancy_grid_history = np.zeros((self.HEIGHT,self.WIDTH,self.IMG_OBS_HIST),dtype=np.bool)
        # TODO rewrite/generalize for history of different instances
        occupancy_grid_history[:,:,0] = np.squeeze(state['occupancy_grid'])
        for t in range(1,self.IMG_OBS_HIST):
            occupancy_grid_history[:,:,t] = np.squeeze(self.prev_states[-t]['occupancy_grid'])

        obs['image'] = np.rollaxis(occupancy_grid_history, 2, 0) # channels as first dim

        #### ADD NOISE TO STUDENT!!!
        state['goal_location'][0] += self.noise_x
        state['goal_location'][1] += self.noise_y
        state['goal_direction'] = self.calc_goal_direction(state)
        state['goal_direction'] = self.calc_goal_direction(state)
        state['global_goal_distance'] = np.linalg.norm(self.goal_location[:2] - state['current_location'][:2])
        #### END

        max_goal_distance = self.grid_config['goal']['max_distance']
        g_dist = min(state['global_goal_distance'],max_goal_distance)/max_goal_distance
        state['goal_distance'] = g_dist
        obs['goal_distance'] = [state['goal_distance']]
        state['normed_goal_direction'] = state['goal_direction']/np.pi
        obs['normed_goal_direction'] = [state['normed_goal_direction']]
        obs['grid_lidar'] = state['grid_lidar'].reshape(-1)
        
        state['obs'] = obs
        return obs, state
    
    def get_info(self,state:OrderedDict)->dict:
        info = dict()
        info['state'] = state
        if self.render_info: info['render_info'] = self.gen_render_info(info)
        return info

    def termination_check(self, state):
        done = False
        success = False

        # terminate if too many violations
        collision_threshold = 1
        non_sidewalk_threshold = self.grid_config['termination']['non_sidewalk_threshold']
        state["collisions_status"] = [self.prev_states[i]['robot_is_colliding'] for i in range(-(collision_threshold-1),0)] + [state['robot_is_colliding']]
        state["non_sidewalks_status"] = [self.prev_states[i]['is_not_on_sidewalk'] for i in range(-(non_sidewalk_threshold-1),0)] + [state['is_not_on_sidewalk']]
        if np.all(state["non_sidewalks_status"]):
            done = True
            success = False
        if np.all(state["collisions_status"]):
            done = True
            success = False
        
        if state['global_goal_distance'] < self.grid_config['termination']['global_goal_distance']:
            done = True
            success = True
        
        state['done'] = done
        state['success'] = success
        return done, state

    def step(self, action):
        self.character.apply_control(*action)

        # for pedestrian in self.pedestrians: # TODO
        #     if self.config['pedestrians']['control_type'] == "random_straight":
        #         d = np.random.uniform(0.3,1.0)
        #         pedestrian.apply_control(0.0,d,0.0)

        self.simulator.step_simulator()

        act_tf = self.character.model.get_transform()
        act_loc = act_tf.location
        act_rot = act_tf.rotation
        sx,sy,sz = act_loc.x, act_loc.y, act_loc.z
        roll,pitch,yaw = act_rot.roll, act_rot.pitch, act_rot.yaw
        sz += 4
        spec_tf = carla.Transform(
            carla.Location(sx,sy,sz),
            carla.Rotation(roll=roll,pitch=-50,yaw=yaw)
            )
        self.spectator.set_transform(spec_tf)
        
        state = self.get_state()
        if self.prev_states is None: self.prev_states = [state] * self.IMG_OBS_HIST #TODO self.config['state_history_size']
        done, state = self.termination_check(state)
        reward, state = 0.0, state # self.get_reward(state) TODO implement 
        obs, state = self.get_observations(state)
        info = self.get_info(state)

        # s(t-1) info tracking        
        self.prev_states.pop(0)
        self.prev_states.append(state) 
        return obs, reward, done, info
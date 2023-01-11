import os
import sys
import glob

import numpy as np

# from common.utils import ranged_angle
# from visenv.sim_world import CARLA_DIR

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

class AbstractAgent:
    def __init__(self, world, model_blueprint, init_location, init_rotation):
        self.height = 0.1
        self.world = world
        self.model_blueprint = model_blueprint
        self.init_transform = carla.Transform()
        self.init_transform.location = init_location
        self.init_transform.rotation = carla.Rotation(*init_rotation)
        self.initialize_model()

    def initialize_model(self):
        self.model = self.world.spawn_actor(self.model_blueprint,self.init_transform)
        # self.model.set_simulate_physics(False)

    def get_xyz_rpy(self):
        transform = self.model.get_transform()
        loc = transform.location
        rot = transform.rotation
        x = loc.x
        y = -loc.y
        z = loc.z
        roll = rot.roll * (np.pi/180)
        pitch = rot.pitch * (np.pi/180)
        yaw = -rot.yaw * (np.pi/180)
        return np.array([x,y,z]), np.array([roll, pitch, yaw])

    def set_xyz_rpy(self, xyz, rpy):
        location = carla.Location()
        rotation = carla.Rotation()
        location.x = xyz[0]
        location.y = -xyz[1]
        location.z = xyz[2]
        rotation.roll = rpy[0] * (180/np.pi)
        rotation.pitch = rpy[1] * (180/np.pi)
        rotation.yaw = -rpy[2] * (180/np.pi)
        
        transform = carla.Transform(location,rotation)
        self.model.set_transform(transform)

    def set_via_carla_transform(self,location,rpy):
        rotation = carla.Rotation()
        rotation.roll = rpy[0]
        rotation.pitch = rpy[1]
        rotation.yaw = rpy[2]
        transform = carla.Transform(location,rotation)
        self.model.set_transform(transform)

    def get_sidewalk_height(self,x,y,z):
        p = self.world.ground_projection(carla.Location(x,-y,z+self.height*2))
        if p is not None and p.label == carla.CityObjectLabel.Sidewalks:
            return p.location.z + self.height
        return z

    def apply_control(self,fwd,dtheta):
        # this control method doesn't handle collisions
        # can be repurposed for something else
        if fwd > 0.0:
            fwd *= 0.2
        else:
            fwd *= 0.1
        dtheta *= 0.3

        xyz, rpy = self.get_xyz_rpy()
        
        facing_theta = rpy[2]
        global_dx = fwd * np.cos(facing_theta)
        global_dy = fwd * np.sin(facing_theta)

        xyz[0] += global_dx
        xyz[1] += global_dy
        xyz[2] = self.get_sidewalk_height(*xyz)
        # ROTATIONS ARE IN DEGREES
        rpy[0] = 0 
        rpy[1] = 0
        rpy[2] = ranged_angle(facing_theta + dtheta)
        # SEND COMMAND
        self.set_xyz_rpy(xyz,rpy)

class GoalAgent:
    def __init__(self, world, model_blueprint, init_location, init_rotation, visible=False):
        self.world = world
        self.model_blueprint = model_blueprint
        self.init_transform = carla.Transform()
        self.init_transform.location = init_location
        self.init_transform.rotation = carla.Rotation(*init_rotation)
        self.visible = visible
        if self.visible: self.initialize_model()

    def initialize_model(self):
        self.model = self.world.spawn_actor(self.model_blueprint,self.init_transform)
        self.model.set_simulate_physics(False)

    def get_xyz_rpy(self):
        # never moves
        loc = self.init_transform.location
        rot = self.init_transform.rotation
        x = loc.x
        y = -loc.y
        z = loc.z
        roll = rot.roll * (np.pi/180)
        pitch = rot.pitch * (np.pi/180)
        yaw = -rot.yaw * (np.pi/180)
        return np.array([x,y,z]), np.array([roll, pitch, yaw])

    def set_via_carla_transform(self,location,rpy):
        rotation = carla.Rotation()
        rotation.roll = rpy[0]
        rotation.pitch = rpy[1]
        rotation.yaw = rpy[2]
        transform = carla.Transform(location,rotation)
        self.init_transform = transform
        if self.visible: self.model.set_transform(transform)
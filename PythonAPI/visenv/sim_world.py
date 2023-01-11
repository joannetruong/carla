import os
import sys
import glob
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

class CarlaSimWorld:
    def __init__(self, world_name, server_port=2000):
        self.server_port = server_port
        self.connect()
        self.world = self.client.get_world()
        self.load_world(world_name)
        self.BPL = self.world.get_blueprint_library()

    def connect(self):
        self.client = carla.Client('localhost', self.server_port)
        self.client.set_timeout(10.0)

    def config_world(self):
        settings = self.world.get_settings()

        settings.no_rendering_mode = False
        settings.synchronous_mode = True
        self.time_step = 1/10
        settings.fixed_delta_seconds = self.time_step #TODO use self.config.FPS

        assert settings.fixed_delta_seconds <= 0.1
        self.world.apply_settings(settings)
        

    def load_world(self, world_name):
        print(f"loading: {world_name}")
        while True:
            try:
                self.world = self.client.load_world(world_name)
                break
            except Exception as e:
                print(f"still waiting load: {world_name}")
                print("exception:",e)
            time.sleep(3)
        print(f"loaded: {world_name}")
        while True:
            try:
                self.config_world()
                break
            except Exception as e:
                print(f"attempting to configure: {world_name}")
                print("exception:",e)
            time.sleep(3)

    def destroy(self,):
        pass

    def step_simulator(self):
        self.world.tick()

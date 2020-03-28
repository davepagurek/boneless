import gym
from gym import spaces
import Box2D
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody)
import math
import numpy as np
from soft_body import SoftBody

class BonelessEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, obj="bodies/quad.obj"):
        super(BonelessEnv, self).__init__()

        self.time = 0

        self.world = world(gravity=(0, -10), doSleep=True)
        self.ground_body = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=polygonShape(box=(50, 1)))

        self.soft_body = SoftBody(self.world, obj)

        self.PPM = 20.0 # pixels per meter
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 480
        self.TARGET_FPS = 30
        self.TIME_STEP = 1.0 / self.TARGET_FPS

        self.viewer = None

        # Actions: [0,1] representing contraction for each joint
        n_edges = self.soft_body.num_edges()
        self.action_space = spaces.Box(
                np.zeros(n_edges),
                np.ones(n_edges),
                dtype=np.float32)

        # TODO
        # Observations: [0,1] representing pressure at each mass?
        n_verts = self.soft_body.num_vertices()
        self.observation_space = spaces.Box(
                np.zeros(n_verts),
                np.ones(n_verts),
                dtype=np.float32)

    def observe_state(self):
        # TODO get state vector from simulation
        return np.zeros(self.soft_body.num_vertices())

    def make_reward(self):
        center_of_mass_x, _ = self.soft_body.get_center_of_mass()
        return center_of_mass_x

    def is_done(self):
        return self.time >= 10

    def reset(self):
        self.time = 0
        self.soft_body.reset()
        observation = self.observe_state()
        return observation

    def step(self, action):
        self.world.Step(self.TIME_STEP, 50, 10)
        self.time += self.TIME_STEP

        observation = self.observe_state()
        reward = self.make_reward()
        done = self.is_done()
        info = None
        # TODO
        return observation, reward, done, {}

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            ground_box = rendering.make_polygon([
                (self.PPM*0, self.PPM*0),
                (self.PPM*50, self.PPM*0),
                (self.PPM*50, self.PPM*1),
                (self.PPM*0, self.PPM*1)
            ])
            self.viewer.add_geom(ground_box)

            self.soft_body.create_geometry(self.viewer, self.PPM)
        
        self.soft_body.update_geometry(self.viewer, self.PPM)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

import gym
from gym import spaces
import Box2D
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody)
import math
import numpy as np
from soft_body import SoftBody

class BonelessEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mesh="bodies/quad.obj", muscles="bodies/muscles.obj"):
        super(BonelessEnv, self).__init__()

        # How much of the previous spring length to preserve when making a change
        # (0 transitions immediately to new spring lengths)
        self.smooth = 0.2

        self.time = 0

        self.world = world(gravity=(0, -10), doSleep=True)
        self.ground_body = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=polygonShape(box=(50, 1)))

        self.soft_body = SoftBody(self.world, mesh, muscles)

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

        # Observations: [0,1] representing compression of each spring
        n_verts = self.soft_body.num_vertices()
        self.observation_space = spaces.Box(
                np.zeros(n_edges),
                np.ones(n_edges),
                dtype=np.float32)

    def observe_state(self):
        n_edges = self.soft_body.num_edges()

        def get_compression(i):
            orig_length = self.soft_body.get_orig_edge_rest_length(i)
            current_length = self.soft_body.get_edge_actual_length(i)
            return current_length / orig_length

        return np.array([ get_compression(i) for i in range(n_edges) ])

    def make_reward(self):
        center_of_mass_x, _ = self.soft_body.get_center_of_mass()
        avg_velocity_x, _ = self.soft_body.get_avg_velocity()
        return center_of_mass_x + avg_velocity_x

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

        for i, scale in enumerate(action):
            scale = min(1, max(0, scale))
            scale = 0.5*scale + 0.5 # Max compression: 0.5
            current_length = self.soft_body.get_edge_rest_length(i)
            orig_length = self.soft_body.get_orig_edge_rest_length(i)
            target_length = scale * orig_length
            new_length = (1 - self.smooth) * target_length + self.smooth * current_length
            self.soft_body.set_edge_rest_length(i, new_length)

        observation = self.observe_state()
        reward = self.make_reward()
        done = self.is_done()
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

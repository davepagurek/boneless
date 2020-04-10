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

        n_edges = self.soft_body.num_edges()
        n_verts = self.soft_body.num_vertices()
        n_muscles = self.soft_body.num_muscles()

        # Actions: Top-left 2x2 submatrix of each muscle's transformation matrix
        self.action_space = spaces.Box(
                -np.finfo(np.float32).max * np.ones(4*n_muscles),
                np.finfo(np.float32).max * np.ones(4*n_muscles),
                dtype=np.float32)

        # Observations: Relative coordinates and velocities of each mass
        self.observation_space = spaces.Box(
                -np.finfo(np.float32).max * np.ones(4*n_verts),
                np.finfo(np.float32).max * np.ones(4*n_verts),
                dtype=np.float32)

    def observe_state(self):
        n_verts = self.soft_body.num_vertices()
        state = []

        com_x, com_y = self.soft_body.get_center_of_mass()
        avg_velocity_x, avg_velocity_y = self.soft_body.get_avg_velocity()
        for i in range(n_verts):
            x, y = self.soft_body.get_vertex_position(i)
            vx, vy = self.soft_body.get_vertex_velocity(i)
            state.append(x - com_x)
            state.append(y - com_y)
            state.append(vx - avg_velocity_x)
            state.append(vy - avg_velocity_y)

        return np.array(state)

    def make_reward(self):
        avg_velocity_x, _ = self.soft_body.get_avg_velocity()
        return avg_velocity_x*10

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

        self.soft_body.set_muscle_transforms(action)
        for i, target in enumerate(self.soft_body.target_edge_lengths()):
            current_length = self.soft_body.get_edge_rest_length(i)
            new_length = (1 - self.smooth) * target + self.smooth * current_length
            self.soft_body.set_edge_rest_length(i, new_length)

        observation = self.observe_state()
        reward = self.make_reward()
        done = self.is_done()

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

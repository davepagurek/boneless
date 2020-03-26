import gym
from gym import spaces
import Box2D
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody)
import math
import statistics
import numpy as np

class BonelessEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, obj="bodies/quad.obj"):
        super(BonelessEnv, self).__init__()

        self.mass_defs = [] # (x, y, r)
        self.joint_defs = [] # (idx_a, idx_b)
        with open(obj) as f:
            for line in f:
                if line.startswith("v "):
                    x, y, _ = [float(loc) for loc in line[2:].split(" ")]
                    self.mass_defs.append((x, y, 1))
                elif line.startswith("f "):
                    indices = [int(idx)-1 for idx in line[2:].split(" ")]
                    for i in range(3):
                        idx_a = indices[i]
                        idx_b = indices[(i+1) % 3]
                        [ax, ay, ar] = self.mass_defs[idx_a]
                        [bx, by, br] = self.mass_defs[idx_b]
                        l = math.hypot(ax-bx, ay-by)
                        ar = min(ar, l*0.5)
                        br = min(br, l*0.5)
                        self.mass_defs[idx_a] = (ax, ay, ar)
                        self.mass_defs[idx_b] = (bx, by, br)
                        self.joint_defs.append((idx_a, idx_b))
        self.total_weight = 10
        self.total_area = sum([ math.pi*r*r for _, _, r in self.mass_defs ])
        self.density = self.total_weight / self.total_area

        self.x0 = 10
        self.y0 = 10
        self.s = 1

        self.joints = []
        self.masses = []
        self.time = 0

        self.world = world(gravity=(0, -10), doSleep=True)
        self.ground_body = self.world.CreateStaticBody(
            position=(0, 0),
            shapes=polygonShape(box=(50, 1)))

        self.PPM = 20.0 # pixels per meter
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 480
        self.TARGET_FPS = 30
        self.TIME_STEP = 1.0 / self.TARGET_FPS

        self.viewer = None

        # Actions: [0,1] representing contraction for each joint
        self.action_space = spaces.Box(
                np.zeros(len(self.joint_defs)),
                np.ones(len(self.joint_defs)),
                dtype=np.float32)

        # TODO
        # Observations: [0,1] representing pressure at each mass?
        self.observation_space = spaces.Box(
                np.zeros(len(self.mass_defs)),
                np.ones(len(self.mass_defs)),
                dtype=np.float32)

    def observe_state(self):
        # TODO get state vector from simulation
        return np.zeros(len(self.mass_defs))

    def make_reward(self):
        center_of_mass_x = statistics.mean([m.position[0] * math.pi * m.fixtures[0].shape.radius**2 * self.density for m in self.masses])
        center_of_mass_x = center_of_mass_x / self.total_weight
        return center_of_mass_x

    def is_done(self):
        return self.time >= 10

    def reset(self):
        for joint in self.joints:
            self.world.DestroyJoint(joint)
        for mass in self.masses:
            self.world.DestroyBody(mass)
        self.joints = []
        self.masses = []
        self.time = 0

        for x, y, r in self.mass_defs:
            body = self.world.CreateDynamicBody(
                    position=(self.x0 + self.s*x, self.y0 + self.s*y),
                    fixedRotation=True)
            circle = body.CreateCircleFixture(
                    radius=r,
                    density=self.density,
                    friction=0.4)
            self.masses.append(body)
        for idx_a, idx_b in self.joint_defs:
            a = self.masses[idx_a]
            b = self.masses[idx_b]
            l = math.hypot(
                    a.position[0]-b.position[0],
                    a.position[1]-b.position[1])
            joint = self.world.CreateDistanceJoint(
                    bodyA=a,
                    bodyB=b,
                    frequencyHz=10.0,
                    dampingRatio=0.5,
                    length=l)
            self.joints.append(joint)

        observation = self.observe_state()
        return observation

    def step(self, action):
        self.world.Step(self.TIME_STEP, 10, 10)
        self.time += self.TIME_STEP

        observation = self.observe_state()
        reward = self.make_reward()
        done = self.is_done()
        info = None
        # TODO
        return observation, reward, done, {}

    def num_joints(self):
        return len(self.joints)

    def get_joint_rest_length(self, idx):
        j = self.joints[idx]
        return j._b2DistanceJoint__GetLength()

    def get_joint_actual_length(self, idx):
        idx_a, idx_b = self.joint_defs[idx]
        a = self.masses[idx_a]
        b = self.masses[idx_b]
        l = math.hypot(
                a.position[0]-b.position[0],
                a.position[1]-b.position[1])
        return l

    def set_joint_rest_length(self, idx, length):
        j = self.joints[idx]
        j._b2DistanceJoint__SetLength(length)

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

            self.transforms = []
            for m in self.masses:
                circle = rendering.make_circle(self.PPM*m.fixtures[0].shape.radius)
                transform = rendering.Transform()
                circle.add_attr(transform)
                self.transforms.append(transform)
                self.viewer.add_geom(circle)

        for transform, mass in zip(self.transforms, self.masses):
            transform.set_translation(self.PPM*mass.position[0], self.PPM*mass.position[1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

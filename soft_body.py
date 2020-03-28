import math
import gym
import numpy as np

class SoftBody:
    def __init__(self, world, path_to_obj):
        self.world = world

        self.mass_defs = [] # (x, y, r)
        self.joint_defs = [] # (idx_a, idx_b)
        with open(path_to_obj) as f:
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
                        ar = max(min(ar, l*0.4), 0.3)
                        br = max(min(br, l*0.4), 0.3)
                        self.mass_defs[idx_a] = (ax, ay, ar)
                        self.mass_defs[idx_b] = (bx, by, br)
                        self.joint_defs.append((idx_a, idx_b))
        self.total_weight = 10
        self.total_area = sum([ math.pi*r*r for _, _, r in self.mass_defs ])
        self.density = self.total_weight / self.total_area
        
        self.x0 = 10
        self.y0 = 10
        self.s = 1 # TODO: rename to scale

        self.joints = []
        self.masses = []

    def reset(self):
        for joint in self.joints:
            self.world.DestroyJoint(joint)
        for mass in self.masses:
            self.world.DestroyBody(mass)
        self.joints = []
        self.masses = []

        for x, y, r in self.mass_defs:
            body = self.world.CreateDynamicBody(
                    position=(self.x0 + self.s*x, self.y0 + self.s*y),
                    fixedRotation=False)
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
                    frequencyHz=20.0,
                    dampingRatio=0.1,
                    length=l)
            self.joints.append(joint)

    def num_vertices(self):
        return len(self.mass_defs)

    def num_edges(self):
        return len(self.joint_defs)

    def get_vertex_position(self, idx):
        m = self.masses[idx]
        return m.position[0], m.position[1]

    def set_vertex_position(self, idx, pos_x, pos_y):
        m = self.masses[idx]
        m.position[0] = pos_x
        m.position[1] = pos_y

    def get_vertex_velocity(self, idx):
        m = self.masses[idx]
        return m.linearVelocity

    def set_vertex_velocity(self, idx, vel_x, vel_y):
        m = self.masses[idx]
        m.linearVelocity[0] = vel_x
        m.linearVelocity[1] = vel_y

    def get_vertex_radius(self, idx):
        return self.masses[idx].fixtures[0].shape.radius

    def set_vertex_radius(self, idx, radius):
        assert(radius > 0.0)
        self.masses[idx].fixtures[0].shape.radius = radius

    def get_edge_rest_length(self, idx):
        j = self.joints[idx]
        return j._b2DistanceJoint__GetLength()

    def get_edge_actual_length(self, idx):
        idx_a, idx_b = self.joint_defs[idx]
        a = self.masses[idx_a]
        b = self.masses[idx_b]
        l = math.hypot(
            a.position[0] - b.position[0],
            a.position[1] - b.position
        )
        return l;

    def set_edge_rest_length(self, idx, length):
        j = self.joints[idx]
        j._b2DistanceJoint__SetLength(length)

    def get_center_of_mass(self):
        def mass(rad):
            return math.pi * rad**2 * self.density

        com_x, com_y = [
            np.mean([m.position[dim] * mass(m.fixtures[0].shape.radius) for m in self.masses]) / self.total_weight
            for dim in [0, 1]
        ]
        return com_x, com_y

    # TODO: allow parent transform to be passed (i.e. for scaling and translating entire view/entire object)
    def create_geometry(self, viewer, PPM):
        self.transforms = []
        from gym.envs.classic_control import rendering
        for m in self.masses:
            rad = m.fixtures[0].shape.radius
            circle = rendering.make_circle(PPM * rad)
            transform = rendering.Transform()
            circle.add_attr(transform)
            self.transforms.append(transform)
            viewer.add_geom(circle)
        
    def update_geometry(self, viewer, PPM):
        for transform, mass in zip(self.transforms, self.masses):
            transform.set_translation(PPM * mass.position[0], PPM * mass.position[1])

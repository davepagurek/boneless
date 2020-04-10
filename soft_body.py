import math
import gym
import numpy as np

from obj import parse_obj

class Muscle:
    def __init__(self, x, y):
        self.position = (x, y)
        self.joint_indices = []
        self.joint_endpoints = []
        self.transform = np.identity(3)
        self.color = [np.random.uniform(), np.random.uniform(), np.random.uniform()]

    def reset(self):
        self.set_transform([1, 0, 0, 1])

    def set_transform(self, transform):
        self.transform = np.array([
            [ transform[0], transform[1], 0],
            [ transform[1], transform[2], 0],
            [0, 0, 1]
        ])

    def get_joint_lengths(self):
        lengths = []
        for idx, ((ax, ay), (bx, by)) in zip(self.joint_indices, self.joint_endpoints):
            relative_a = np.array([ax-self.position[0], ay-self.position[1], 1]).T
            relative_b = np.array([bx-self.position[0], by-self.position[1], 1]).T

            transformed_a = self.transform @ relative_a
            transformed_b = self.transform @ relative_b
            target_length = np.linalg.norm(transformed_a - transformed_b)
            orig_length = math.hypot(ax-bx, ay-by);
            target_length = max(0.25*orig_length, min(orig_length, target_length))
            lengths.append((idx, target_length))

        return lengths

    def add_joint(self, joint_idx, endpoint_a, endpoint_b):
        self.joint_indices.append(joint_idx)
        self.joint_endpoints.append((endpoint_a, endpoint_b))

    def dist_to(self, x, y):
        return math.hypot(self.position[0] - x, self.position[1] - y)

class SoftBody:
    # Assumes muscles are in x,y plane. In blender, export an obj with x forward, z up
    def __init__(self, world, mesh_path, muscles_path):
        self.world = world

        self.mass_defs = [] # (x, y, r)
        self.joint_defs = [] # (idx_a, idx_b)

        mass_verts, mass_edges = parse_obj(mesh_path)
        muscle_verts, _ = parse_obj(muscles_path)

        self.muscles = [ Muscle(x,y) for x, y, _ in muscle_verts ]
        self.mass_defs = [ (x, y, 1) for x, y, _ in mass_verts ] # (x, y, r)
        self.joint_defs = mass_edges # (idx_a, idx_b)
        self.joint_muscles = []

        for joint_idx, (idx_a, idx_b) in enumerate(self.joint_defs):
            [ax, ay, ar] = self.mass_defs[idx_a]
            [bx, by, br] = self.mass_defs[idx_b]

            # Update mass radii
            l = math.hypot(ax-bx, ay-by)
            ar = max(min(ar, l*0.4), 0.3)
            br = max(min(br, l*0.4), 0.3)
            self.mass_defs[idx_a] = (ax, ay, ar)
            self.mass_defs[idx_b] = (bx, by, br)

            # Assign this joint to closest muscle to joint center
            dists = [ m.dist_to((ax+bx)/2, (ay+by)/2) for m in self.muscles ]
            closest_muscle = dists.index(min(dists))

            self.muscles[closest_muscle].add_joint(joint_idx, (ax, ay), (bx, by))
            self.joint_muscles.append(self.muscles[closest_muscle])

        self.total_weight = 10
        self.total_area = sum([ math.pi*r*r for _, _, r in self.mass_defs ])
        self.density = self.total_weight / self.total_area
        
        self.x0 = 10
        self.y0 = 5
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
        for muscle in self.muscles:
            muscle.reset()

    def num_vertices(self):
        return len(self.mass_defs)

    def num_edges(self):
        return len(self.joint_defs)

    def num_muscles(self):
        return len(self.muscles)

    def set_muscle_transforms(self, transforms):
        for transform, muscle in zip(np.split(transforms, self.num_muscles()), self.muscles):
            muscle.set_transform(transform)

    def target_edge_lengths(self):
        targets = [1] * self.num_edges()
        for muscle in self.muscles:
            for joint_idx, length in muscle.get_joint_lengths():
                targets[joint_idx] = length
        return targets

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

    def get_orig_edge_rest_length(self, idx):
        idx_a, idx_b = self.joint_defs[idx]
        ax, ay, ar = self.mass_defs[idx_a]
        bx, by, br = self.mass_defs[idx_b]
        return math.hypot(self.s*(ax-bx), self.s*(ay-by))

    def get_edge_actual_length(self, idx):
        idx_a, idx_b = self.joint_defs[idx]
        a = self.masses[idx_a]
        b = self.masses[idx_b]
        l = math.hypot(
            a.position[0] - b.position[0],
            a.position[1] - b.position[1]
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

    def get_avg_velocity(self):
        def mass(rad):
            return math.pi * rad**2 * self.density

        v_x, v_y = [
            np.mean([m.linearVelocity[dim] * mass(m.fixtures[0].shape.radius) for m in self.masses]) / self.total_weight
            for dim in [0, 1]
        ]
        return v_x, v_y

    # TODO: allow parent transform to be passed (i.e. for scaling and translating entire view/entire object)
    def create_geometry(self, viewer, PPM):
        self.transforms = []
        self.lines = []
        from gym.envs.classic_control import rendering
        for m in self.masses:
            rad = m.fixtures[0].shape.radius
            circle = rendering.make_circle(PPM * rad)
            transform = rendering.Transform()
            circle.add_attr(transform)
            self.transforms.append(transform)
            viewer.add_geom(circle)

        for muscle, (idx_a, idx_b) in zip(self.joint_muscles, self.joint_defs):
            line = rendering.make_polyline([(0, 0), (1, 0)])
            line.set_linewidth(2)
            line.set_color(*muscle.color)
            viewer.add_geom(line)
            self.lines.append(line)
        
    def update_geometry(self, viewer, PPM):
        for transform, mass in zip(self.transforms, self.masses):
            transform.set_translation(PPM * mass.position[0], PPM * mass.position[1])
        for line, (idx_a, idx_b) in zip(self.lines, self.joint_defs):
            mass_a = self.masses[idx_a]
            mass_b = self.masses[idx_b]
            line.v[0] = (PPM * mass_a.position[0], PPM * mass_a.position[1])
            line.v[1] = (PPM * mass_b.position[0], PPM * mass_b.position[1])

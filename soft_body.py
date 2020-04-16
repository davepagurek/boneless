import math
import gym
import numpy as np
import itertools
from gym.envs.classic_control import rendering

from obj import parse_obj

import pyglet
from pyglet.gl import *

class TexturedTriangles(rendering.Geom):
    def __init__(self, texture_path):
        rendering.Geom.__init__(self)
        self.img = pyglet.image.load(texture_path)
        self.tex = self.img.get_texture()

    def set_data(self, vertices, uvs, indices):
        self.vertices = []
        for x, y in vertices:
            self.vertices.append(x)
            self.vertices.append(y)
            self.vertices.append(0)
        self.vertices = tuple(self.vertices)

        self.uvs = []
        for u, v in uvs:
            self.uvs.append(u)
            self.uvs.append(v)
        self.uvs = tuple(self.uvs)

        self.indices = []
        for a, b, c in indices:
            self.indices.append(a)
            self.indices.append(b)
            self.indices.append(c)

    def render1(self):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.tex.id)
        glColor4f(1, 1, 1, 1)
        pyglet.graphics.draw_indexed(
                len(self.vertices) // 3,
                GL_TRIANGLES,
                self.indices,
                ('v3f', self.vertices),
                ('t2f', self.uvs))

class Muscle:
    def __init__(self, x, y):
        self.position = (x, y)
        self.joint_indices = []
        self.joint_endpoints = []
        self.transform = np.identity(3)
        self.color = [np.random.uniform(), np.random.uniform(), np.random.uniform()]

    def reset(self):
        self.set_transform([0, 0, 1])

    def set_transform(self, transform):
        assert(len(transform) == 3)
        angle = transform[0]
        skew = transform[1]
        scale = 0.1 + 1 / (1 + math.exp(-transform[2]))
        # scale = 1/(1+abs(skew))
        rotation = np.array([
            [ math.cos(angle), -math.sin(angle), 0],
            [ math.sin(angle), math.cos(angle), 0],
            [ 0, 0, 1 ]
        ])
        shear = np.array([
            [ scale, skew, 0 ],
            [ 0, scale, 0 ],
            [ 0, 0, 1 ]
        ])
        self.transform = shear @ rotation

    def get_joint_lengths(self):
        lengths = []
        for idx, ((ax, ay), (bx, by)) in zip(self.joint_indices, self.joint_endpoints):
            relative_a = np.array([ax-self.position[0], ay-self.position[1], 1]).T
            relative_b = np.array([bx-self.position[0], by-self.position[1], 1]).T

            transformed_a = self.transform @ relative_a
            transformed_b = self.transform @ relative_b
            target_length = np.linalg.norm(transformed_a - transformed_b)
            orig_length = math.hypot(ax-bx, ay-by)
            target_length = max(0.5*orig_length, min(2*orig_length, target_length))
            lengths.append((idx, target_length))

        return lengths

    def add_joint(self, joint_idx, endpoint_a, endpoint_b):
        self.joint_indices.append(joint_idx)
        self.joint_endpoints.append((endpoint_a, endpoint_b))

    def get_joint_indices(self):
        return self.joint_indices

    def dist_to(self, x, y):
        return math.hypot(self.position[0] - x, self.position[1] - y)

class SoftBody:
    # Assumes muscles are in x,y plane. In blender, export an obj with x forward, z up
    def __init__(self, world, mesh_path, muscles_path):
        self.world = world

        self.mass_defs = [] # (x, y, r)
        self.joint_defs = [] # (idx_a, idx_b)

        mass_verts, mass_edges, mass_faces = parse_obj(mesh_path)
        muscle_verts, _, _ = parse_obj(muscles_path)

        min_x = min([ x for x, _, _ in mass_verts ])
        max_x = max([ x for x, _, _ in mass_verts ])
        min_y = min([ y for _, y, _ in mass_verts ])
        max_y = max([ y for _, y, _ in mass_verts ])

        self.uvs = [ ((x-min_x)/(max_x-min_x), (y-min_y)/(max_y-min_y)) for x, y, _ in mass_verts ]

        self.muscles = [ Muscle(x,y) for x, y, _ in muscle_verts ]

        # array with same indices as self.muscles, each entry is the list
        # of adjacent muscles for that muscle
        self.muscle_adjacencies = [ [] for _ in muscle_verts ]

        self.mass_defs = [ (x, y, 1.0) for x, y, _ in mass_verts ] # (x, y, r)
        self.joint_defs = mass_edges # (idx_a, idx_b)
        self.joint_muscles = []
        self.mass_faces = mass_faces

        # array with same indices as mass defs, each element is the muscle(s)
        # to which that mass belongs
        muscles_per_mass = [ [] for _ in mass_verts ]

        def add_muscle_at_mass(muscle_idx, mass_idx):
            lst = muscles_per_mass[mass_idx]
            if not muscle_idx in lst:
                lst.append(muscle_idx)


        for joint_idx, (idx_a, idx_b) in enumerate(self.joint_defs):
            [ax, ay, ar] = self.mass_defs[idx_a]
            [bx, by, br] = self.mass_defs[idx_b]

            # Update mass radii
            l = math.hypot(ax-bx, ay-by)
            ar = max(min(ar, l*0.2), 0.3)
            br = max(min(br, l*0.2), 0.3)
            self.mass_defs[idx_a] = (ax, ay, ar)
            self.mass_defs[idx_b] = (bx, by, br)

            # Assign this joint to closest muscle to joint center
            dists = [ m.dist_to((ax+bx)/2, (ay+by)/2) for m in self.muscles ]
            closest_muscle = dists.index(min(dists))

            self.muscles[closest_muscle].add_joint(joint_idx, (ax, ay), (bx, by))
            self.joint_muscles.append(self.muscles[closest_muscle])

            # record which muscles influence the vertices
            add_muscle_at_mass(closest_muscle, idx_a)
            add_muscle_at_mass(closest_muscle, idx_b)

        self.total_weight = 10
        self.total_area = sum([ math.pi*r*r for _, _, r in self.mass_defs ])
        self.density = self.total_weight / self.total_area
        
        self.x0 = 10
        self.y0 = 5
        self.s = 1 # TODO: rename to scale

        self.joints = []
        self.masses = []

        # for each mass and the muscles that influence it directly
        for mass_idx, influencing_muscles in enumerate(muscles_per_mass):
            # for each pair of muscles
            for (muscle_a, muscle_b) in itertools.combinations(influencing_muscles, 2):
                adjacent_to_a = self.muscle_adjacencies[muscle_a]
                adjacent_to_b = self.muscle_adjacencies[muscle_b]
                if not muscle_a in adjacent_to_b:
                    adjacent_to_b.append(muscle_a)
                if not muscle_b in adjacent_to_a:
                    adjacent_to_a.append(muscle_b)

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
                    fixedRotation=True)
            circle = body.CreateCircleFixture(
                    radius=r,
                    density=self.density,
                    friction=0.6)
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
                    frequencyHz=16.0,
                    dampingRatio=0.5,
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

    def get_muscle(self, idx):
        return self.muscles[idx]

    def get_muscle_edge_actual_lengths(self, muscle_idx):
        ret = []
        for j, m in enumerate(self.joint_muscles):
            if (m == muscle_idx):
                ret.append(self.get_edge_actual_length(j))
        return ret
                

    def set_muscle_transforms(self, transforms):
        assert(isinstance(transforms, np.ndarray))
        assert(len(transforms.shape) == 1)
        assert(transforms.shape[0] == (self.num_muscles() * 3))
        for transform, muscle in zip(np.split(transforms, self.num_muscles()), self.muscles):
            muscle.set_transform(transform)

    def get_adjacent_muscles(self, idx):
        return self.muscle_adjacencies[idx]

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
        return l

    def set_edge_rest_length(self, idx, length):
        j = self.joints[idx]
        length = min(max(length, 0.5), 1.0)
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
        self.shape = TexturedTriangles("textures/creature.png")
        viewer.add_geom(self.shape)
        # self.transforms = []
        # self.lines = []
        # for m in self.masses:
            # rad = m.fixtures[0].shape.radius
            # circle = rendering.make_circle(PPM * rad)
            # transform = rendering.Transform()
            # circle.add_attr(transform)
            # self.transforms.append(transform)
            # viewer.add_geom(circle)

        # for muscle, (idx_a, idx_b) in zip(self.joint_muscles, self.joint_defs):
            # line = rendering.make_polyline([(0, 0), (1, 0)])
            # line.set_linewidth(2)
            # line.set_color(*muscle.color)
            # viewer.add_geom(line)
            # self.lines.append(line)
        
    def update_geometry(self, viewer, PPM):
        vertices = [ [PPM*m.position[0], PPM*m.position[1]] for m in self.masses ]
        self.shape.set_data(vertices, self.uvs, self.mass_faces)
        # for transform, mass in zip(self.transforms, self.masses):
            # transform.set_translation(PPM * mass.position[0], PPM * mass.position[1])
        # for line, (idx_a, idx_b) in zip(self.lines, self.joint_defs):
            # mass_a = self.masses[idx_a]
            # mass_b = self.masses[idx_b]
            # line.v[0] = (PPM * mass_a.position[0], PPM * mass_a.position[1])
            # line.v[1] = (PPM * mass_b.position[0], PPM * mass_b.position[1])

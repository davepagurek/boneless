import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import Box2D
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody)
import math

PPM = 20.0 # pixels per meter
TARGET_FPS = 30
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Boneless")
clock = pygame.time.Clock()

world = world(gravity=(0, -10), doSleep=True)
ground_body = world.CreateStaticBody(
    position=(0, 0),
    shapes=polygonShape(box=(50, 1)),
)

x0 = 10
y0 = 10
s = 1
total_weight = 10

masses = []
joints = []

min_dists = []

with open("bodies/quad.obj") as f:
    for line in f:
        if line.startswith("v "):
            x, y, _ = [float(loc) for loc in line[2:].split(" ")]
            body = world.CreateDynamicBody(
                    position=(x0 + s*x, y0 + s*y),
                    fixedRotation=True)
            circle = body.CreateCircleFixture(
                    radius=0.3,
                    density=1,
                    friction=0.4)
            masses.append(body)
            min_dists.append(1)
        elif line.startswith("f "):
            indices = [int(idx)-1 for idx in line[2:].split(" ")]
            for i in range(3):
                a = masses[indices[i]]
                b = masses[indices[(i+1) % 3]]
                l = math.hypot(a.position[0]-b.position[0], a.position[1]-b.position[1])
                min_dists[indices[i]] = min(l, min_dists[indices[i]])
                min_dists[indices[(i+1) % 3]] = min(l, min_dists[indices[(i+1) % 3]])
                joint = world.CreateDistanceJoint(
                        bodyA=a,
                        bodyB=b,
                        frequencyHz = 10.0,
                        dampingRatio=0.5,
                        length=l)
                joints.append(joint)

total_area = 0
for i, body in enumerate(masses):
    body.fixtures[0].shape.radius = min_dists[i] / 2
    total_area += math.pi * body.fixtures[0].shape.radius**2

for body in masses:
    body.fixtures[0].density = total_weight /total_area 

def draw_polygon(polygon, body, fixture):
    vertices = [(body.transform * v) * PPM for v in polygon.vertices]
    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    pygame.draw.polygon(screen, (255, 255, 255, 255), vertices)
polygonShape.draw = draw_polygon

def draw_circle(circle, body, fixture):
    position = body.transform * circle.pos * PPM
    position = (position[0], SCREEN_HEIGHT - position[1])
    pygame.draw.circle(
            screen,
            (255, 255, 255, 255),
            [int(x) for x in position],
            int(circle.radius * PPM))
circleShape.draw = draw_circle

running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            running = False

    screen.fill((0, 0, 0, 255))

    for body in world.bodies:
       for fixture in body.fixtures:
           fixture.shape.draw(body, fixture)

    world.Step(TIME_STEP, 10, 10)

    pygame.display.flip()
    clock.tick(TARGET_FPS)

pygame.quit()
print('Done!')

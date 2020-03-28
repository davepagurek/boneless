from boneless_env import BonelessEnv
import time

env = BonelessEnv()

obs = env.reset()
done = False
t = 0
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    if t == 100:
        for i in range(env.soft_body.num_edges()):
            env.soft_body.set_edge_rest_length(i, 1.0)

    env.render()
    t += 1
    time.sleep(1.0/30.0)

env.close()

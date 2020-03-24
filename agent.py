from boneless_env import BonelessEnv
import time

env = BonelessEnv()

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    env.render()
    time.sleep(1.0/30.0)

env.close()

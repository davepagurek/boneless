from boneless_env import BonelessEnv
# from simple_neural_network import SimpleNeuralNetwork
from soft_body_controller import SoftBodyController

import sys
import torch
import numpy as np
import random
import threading
from multiprocessing import Pool
import pickle
import time

def make_new_env():
    # return BonelessEnv("bodies/skinnyworm.obj", "bodies/skinnyworm-muscles.obj")
    # return BonelessEnv("bodies/fatworm.obj", "bodies/fatworm-muscles.obj")
    return BonelessEnv("bodies/quad.obj", "bodies/quad-muscles.obj")
    # return BonelessEnv("bodies/horse.obj", "bodies/horse-muscles.obj")

hidden_state_dim = 8

global_env = make_new_env()

obs_dim = global_env.observation_space.shape[0]
act_dim = global_env.action_space.shape[0]

def make_new_controller(env):
    return SoftBodyController(env.get_soft_body(), hidden_state_dim)
    # return SimpleNeuralNetwork(obs_dim, act_dim)

def mutate_policy_vector(policy, sigma):
    return policy + sigma * (2.0 * np.random.random(policy.shape) - 1.0)


pi_dim = make_new_controller(global_env).num_parameters()

def create_policy(env, policy_vector):
    pi = make_new_controller(env)
    pi.replace_parameters(policy_vector)
    return pi

def initial_policy():
    return mutate_policy_vector(np.zeros(pi_dim), 1.0)

def eval_policy(env, policy_vector, render=False):
    pi = create_policy(env, policy_vector)
    obs = env.reset()
    total_reward = 0
    while True:
        act = pi.step(obs)
        obs, reward, done, _ = env.step(act)
        total_reward += reward
        if render:
            env.render()
        if done:
            break
    # if render:
    #     env.close()
    return total_reward

def eval_policy_process_worker(pi):
    local_env = make_new_env()
    r = eval_policy(local_env, pi)
    return (pi, r)

def main():
    # population size
    N = 200

    # the population
    P = [initial_policy() for _ in range(N)]

    # truncation selection
    T = 10

    # mutation rate
    sigma = 0.001

    # just for visualizing the first random policy
    # eval_policy(global_env, P[0], render=True)

    population_progress = []

    current_generation = 0
    for i in range(500):
        current_generation += 1
        print("Generation", current_generation)

        start_time = time.time()

        with Pool() as process_pool:
            # P_and_rewards = process_pool.map(eval_policy_process_worker, P)
            P_and_rewards = []
            for i, res in enumerate(process_pool.imap_unordered(eval_policy_process_worker, P, 1)):
                P_and_rewards.append(res)
                sys.stdout.write("\r[%-50s] %d/%d" % ('=' * ((i + 1) * 50 // N), (i + 1), N))
                sys.stdout.flush()
        print("\n")

        end_time = time.time()

        take_first = lambda x: x[0]
        take_second = lambda x: x[1]
        P_and_rewards_sorted = sorted(P_and_rewards, key=take_second)
        top_T = list(map(take_first, P_and_rewards_sorted[-T:]))
        best_pi = top_T[-1]

        population_progress.append(list(map(take_second, P_and_rewards_sorted)))

        P.clear()
        P.append(best_pi)
        while len(P) < N:
            parent_pi = random.choice(top_T)
            new_pi = mutate_policy_vector(parent_pi, sigma)
            P.append(new_pi)

        print("    Top reward:", P_and_rewards_sorted[-1][1])
        top_T_rewards = list(map(take_second, P_and_rewards_sorted[-T:]))
        print("    Average reward of top", T, "policies:", np.mean(np.array(top_T_rewards)))
        print("    Population execution time: {0} seconds".format((end_time - start_time)))
        print("\n")
        with open("best_policy_vector_so_far.pkl", "wb") as outfile:
            pickle.dump(best_pi, outfile)
        with open("population_progress.pkl", "wb") as outfile:
            pickle.dump(np.array(population_progress), outfile)

        if current_generation % 2 == 0:
            eval_policy(global_env, best_pi, render=True)

if __name__ == "__main__":
    main()

# To visualize a saved policy, run this instead of main:
# with open("best_policy_vector_so_far.pkl", "rb") as infile:
#     best_pi = pickle.load(infile)
# eval_policy(global_env, best_pi, render=True)
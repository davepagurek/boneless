from boneless_env import BonelessEnv
from simple_neural_network import SimpleNeuralNetwork

import sys
import torch
import numpy as np
import random
import threading
from multiprocessing import Pool
import pickle
import time

global_env = BonelessEnv()

obs_dim = global_env.observation_space.shape[0]
act_dim = global_env.action_space.shape[0]

def mutate_policy_vector(policy, sigma):
    return policy + sigma * (2.0 * np.random.random(policy.shape) - 1.0)

pi_dim = SimpleNeuralNetwork(obs_dim, act_dim).num_parameters()

def create_policy(policy_vector):
    pi = SimpleNeuralNetwork(obs_dim, act_dim)
    pi.replace_parameters(policy_vector)
    return pi

def initial_policy():
    return mutate_policy_vector(np.zeros(pi_dim), 2.0)

def eval_policy(env, policy_vector, render=False):
    pi = create_policy(policy_vector)
    obs = env.reset()
    total_reward = 0
    while True:
        act = pi(torch.tensor(obs)).detach().numpy()
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
    local_env = BonelessEnv()
    r = eval_policy(local_env, pi)
    return (pi, r)

def main():
    # population size
    N = 100

    # the population
    P = [initial_policy() for _ in range(N)]

    # truncation selection
    T = 10

    # mutation rate
    sigma = 0.1

    # just for visualizing the first random policy
    eval_policy(global_env, P[0], render=True)

    current_generation = 0
    while True:
        current_generation += 1
        print("Generation", current_generation)

        start_time = time.time()

        with Pool() as process_pool:
            P_and_rewards = process_pool.map(eval_policy_process_worker, P)

        end_time = time.time()

        take_first = lambda x: x[0]
        take_second = lambda x: x[1]
        P_and_rewards_sorted = sorted(P_and_rewards, key=take_second)
        top_T = list(map(take_first, P_and_rewards_sorted[-T:]))
        best_pi = top_T[-1]

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

        if current_generation % 10 == 0:
            eval_policy(global_env, best_pi, render=True)

if __name__ == "__main__":
    main()
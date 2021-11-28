import os
import time
import random
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Network
import environment
import config
import agent




def test(env_name=config.env_name,
         save_interval=config.save_interval,
         test_epsilon=config.test_epsilon,
         save_plot=config.save_plot,
         num_actions = config.num_actions):
    env = environment.create_env(mode="test")
    test_round = 5
    x, y = [], []

    network = Network(num_actions)
    checkpoint = save_interval
    while os.path.exists('./models/{}-{}.pth'.format(env_name, checkpoint)):
        x.append(checkpoint)
        network.load_state_dict(torch.load('./models/{}-{}.pth'.format(env_name, checkpoint)))
        sum_reward = 0
        for _ in range(test_round):
            ag = agent.SmartAgent(network, test_epsilon)
            obs = env.reset()
            done = False
            while not done:
                action_no, action, q_val, rew = ag.step(obs[0])
                actions = [action]
                obs = env.step(actions)
                reward = rew
                # make reward obvious
                done = obs[0].last()
                sum_reward += reward

        print(' checkpoint: {}'.format(checkpoint))
        print(' average reward: {}\n'.format(sum_reward / test_round))
        y.append(sum_reward / test_round)
        checkpoint += save_interval

    plt.title(env_name)
    plt.xlabel('training steps')
    plt.ylabel('average reward')

    plt.plot(x, y)

    if save_plot:
        plt.savefig('./{}.jpg'.format(env_name))
    plt.show()


if __name__ == '__main__':
    test()



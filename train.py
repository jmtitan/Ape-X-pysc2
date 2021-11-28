import random
import time
import torch
import numpy as np
from worker import Learner, Actor, ReplayBuffer
import config
import ray
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_epsilon(actor_id: int, base_eps: float = config.base_eps, alpha: float = config.alpha,
                num_actors: int = config.num_actors):
    exponent = 1 + actor_id / (num_actors - 1) * alpha
    return base_eps ** exponent


def train(num_actors=config.num_actors, log_interval=config.log_interval):
    ray.init()
    buffer = ReplayBuffer.remote()
    learner = Learner.remote(buffer)
    actors = [Actor.remote(get_epsilon(i), learner, buffer) for i in range(num_actors)]

    for actor in actors:
        actor.run.remote()

    while not ray.get(buffer.ready.remote()):
        time.sleep(log_interval)
        ray.get(learner.stats.remote(log_interval))
        print()

    print('\n\n\nstart training\n\n\n')
    buffer.run.remote()
    learner.run.remote()

    done = False
    while not done:
        time.sleep(log_interval)
        done = ray.get(learner.stats.remote(log_interval))
        print()

    # buffer = ReplayBuffer()
    # learner = Learner(buffer)
    # actor = Actor(0.9, learner, buffer)
    #
    #
    # background_thread = threading.Thread(target=actor.run, daemon=True)
    # background_thread.start()
    #
    # while not (buffer.ready()):
    #     time.sleep(log_interval)
    #     learner.stats(log_interval)
    #     print()
    #
    # print('\n\n\nstart training\n\n\n')
    # buffer.run()
    # learner.run()
    #
    # done = False
    # while not done:
    #     time.sleep(log_interval)
    #     done = learner.stats(log_interval)
    #     print()


if __name__ == '__main__':
    train()


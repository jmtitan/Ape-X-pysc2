import sys
from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import features
import myagent
from run_myloop import run_loop
import multiprocessing as mp
import time
import numpy as np
FLAGS = flags.FLAGS

     
def main(core_number=1):
  FLAGS(sys.argv)
  with sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.terran, "Tergot"),
                     sc2_env.Bot(sc2_env.Race.random,
                                 sc2_env.Difficulty.very_easy)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=86, minimap=64),
                use_feature_units=True),
            step_mul=4,
            realtime=False,
            save_replay_episodes=0,
            visualize=False) as env:
    agent=myagent.RandomAgent()
    run_loop([agent], env, max_frames=int(1000/core_number))


if __name__ =='__main__':
    main()





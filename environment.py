from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import actions

import config

def create_env(mapname = config.env_name, px1 = config.screen, px2 = config.minimap, mode = 'train'):
    FLAGS = flags.FLAGS
    FLAGS([__file__])
    if mode == "train":
        env = sc2_env.SC2Env(
            map_name= mapname,
            players= [sc2_env.Agent(sc2_env.Race.terran,"Tergot"),
                    sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
            step_mul= 4,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen = px1, minimap = px2),
                use_feature_units=True),
                realtime= False,
                # save_replay_episodes=0,
                visualize=False
        )
    else:
        env = sc2_env.SC2Env(
            map_name= mapname,
            players= [sc2_env.Agent(sc2_env.Race.terran,"Tergot"),
                    sc2_env.Bot(sc2_env.Race.random,
                                sc2_env.Difficulty.very_easy)],
            step_mul= 4,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen = px1, minimap = px2),
                use_feature_units=True),
                realtime= False,
                save_replay_episodes=1000,
                visualize=False,
                replay_dir='/home/tryit/pysc2/tjm/Ape-X/replay'
                )
    return env

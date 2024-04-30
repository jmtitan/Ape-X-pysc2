from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import memorybuffer
import time
buffer = memorybuffer.Memory_Buffer()


def run_loop(agents, env, max_frames=0, max_episodes=0):
  total_frames = 0
  total_episodes = 0
  start_time = time.time()

  observation_spec = env.observation_spec()
  action_spec = env.action_spec()
  for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
    agent.setup(obs_spec, act_spec)

  try:

    while not max_episodes or total_episodes < max_episodes:
      total_episodes += 1
      obs = env.reset()
      for a in agents:
        a.reset()
      while True:
        total_frames += 1
        actions = [agent.step(timestep)
                   for agent, timestep in zip(agents, obs)]
        if max_frames and total_frames >= max_frames:
          return
        if obs[0].last():
          break
        previous_obs = obs[0].observation
        reward = obs[0].reward
        obs = env.step(actions)
        buffer.push(previous_obs, actions, reward, obs[0].observation)
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f fps" % (
        elapsed_time, total_frames, total_frames / elapsed_time))
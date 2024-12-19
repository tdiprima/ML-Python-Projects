"""
Visually renders 10 episodes of the predefined environment ("CliffWalking-v0") in the OpenAI Gym library
with random actions at each step, displaying the number of timesteps at the end of each episode.
"""
import time

import gym
print(gym.__version__)

# name = "FrozenLake-v1"
# name = "LunarLander-v2"
# name = "CartPole-v1"
name = "CliffWalking-v0"

env = gym.make(name, render_mode="human")
env.reset()
env.render()

for episode in range(10):
    env.reset()
    print("Episode finished after {} timesteps".format(episode))
    for ik in range(100):
        env.render()
        observation, reward, done, _, info = env.step(env.action_space.sample())
        if done:
            break
        time.sleep(0.02)

env.close()

# TODO: Calculate accuracy - Lunar Lander seems to work!

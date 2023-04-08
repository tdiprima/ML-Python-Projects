import time

import gym

# name = 'FrozenLake-v1'  # TODO: LOL do this one manually.
# name = 'LunarLander-v2'  # This works the best!
name = 'CartPole-v1'  # idk wth it's doing, but it works lol

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

# CartPole
# Environment checker?

import gym
# from gym.utils.env_checker import check_env  # OOPS
from gym.utils import env_checker

env9 = gym.make("CartPole-v1", render_mode="human")

raw_env = env9.unwrapped

lulu = env_checker.check_env(raw_env)

# lulu = check_env(env9)
print(f"Type: {type(lulu)}, Thing: {lulu}")

env9.close()

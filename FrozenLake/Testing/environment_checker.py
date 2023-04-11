# CartPole
# Environment checker

import gym
from gym.utils import env_checker

env = gym.make("CartPole-v1", render_mode="human")
print("\nenv:", env)

raw_env = env.unwrapped
print("\nraw_env:", raw_env)


result = env_checker.check_env(raw_env)
print("\nenv_checker.check_env(raw_env):\n", result)

print(f"\nType: {type(result)}\nThing: {result}")

env.close()

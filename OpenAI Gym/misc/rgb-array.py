# CartPole - RGB array
import sys

import gym

try:
    env2 = gym.make("CartPole-v1", render_mode="rgb_array")
    observation = env2.reset()
    rendered = env2.render()
    print(f"Observation: {observation}\nRendered: {rendered}")
    env2.close()
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print(exc_type)
    print(exc_obj)
    print("Line:", exc_tb.tb_lineno)

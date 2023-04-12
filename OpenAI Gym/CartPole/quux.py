# Buddy thought I was trying to get an rgb_array for some reason.
# So they recommended this.
import sys

import gym
from gym import wrappers

try:
    env4 = gym.make("CartPole-v0", render_mode="human")
    env4 = wrappers.Monitor(env4, "./gym-results", force=True)
    observation = env4.reset()

    # Render the environment and get an RGB array
    viewer = env4.render(mode='rgb_array')
    screen = viewer.get_array()

    print(screen.shape)  # should output (400, 600, 3)

except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print(exc_type)
    print(exc_obj)
    print("Line:", exc_tb.tb_lineno)

# CartPole
# import gymgrid
import sys

import gym

try:
    from gym.envs.classic_control import rendering

    env7 = gym.make("CartPole-v0", render_mode="human")
    observation = env7.reset()

    # Create a rendering window and set it as the active context
    viewer = rendering.SimpleImageViewer()
    viewer.window.set_as_current()

    # Render the environment and get an RGB array
    env7.render()
    screen = viewer.get_image()

    print(screen.shape)  # should output (400, 600, 3)

    # Close the rendering window
    viewer.close()
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print(exc_type)
    print(exc_obj)
    print("Line:", exc_tb.tb_lineno)

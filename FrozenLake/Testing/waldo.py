"""
https://stackoverflow.com/questions/71973392/importerror-cannot-import-rendering-from-gym-envs-classic-control
https://www.gymlibrary.dev/content/basic_usage/
https://github.com/openai/gym/issues/2795
Lunar Lander - box2d
pip install box2d fails
pip install box2d pygame fails
pip install gym[box2d]
zsh: no matches found: gym[box2d]
OK, so switch to bash
pip install gym[box2d] pygame
At least it finished, but with warnings and stuff.
Still couldn't get the darn thing to work, even running python from bash.
Solution:
Simply don't import box2d.
"""
import sys

import gym

try:
    import box2d

    env8 = gym.make("LunarLander-v2", render_mode="human")
    env8.action_space.seed(42)

    observation, info = env8.reset(seed=42)

    for _ in range(1000):
        observation, reward, terminated, truncated, info = env8.step(env8.action_space.sample())

        if terminated or truncated:
            observation, info = env8.reset()

    env8.close()

except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print(exc_type)
    print(exc_obj)
    print("Line:", exc_tb.tb_lineno)

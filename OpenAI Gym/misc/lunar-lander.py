"""
Creates a LunarLander-v2 environment using OpenAI Gym, performs 1000 random actions
in that environment, and resets the environment when it terminates or truncates.

Compare to do_1000
https://www.gymlibrary.dev/content/basic_usage/
"""
import gym

env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()

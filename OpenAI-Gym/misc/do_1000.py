"""
Compare to Lunar Lander
Uses the gymnasium library to run a simulation of the game Space Invaders, taking
random actions, and resetting the environment when a game ends.
https://gymnasium.farama.org/content/basic_usage/
"""
import gymnasium as gym

# env = gym.make("LunarLander-v2", render_mode="human")
# env = gym.make("CarRacing-v2", render_mode="human")
# env = gym.make("ALE/MsPacman-v5", render_mode="human")
# env = gym.make("ALE/Pitfall-v5", render_mode="human")
# env = gym.make("ALE/Adventure-v5", render_mode="human")
# env = gym.make("ALE/Defender-v5", render_mode="human")
# env = gym.make("ALE/Qbert-v5", render_mode="human")
env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
# env = gym.make("ALE/YarsRevenge-v5", render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    # print(observation, reward, terminated, truncated, info)

    if terminated or truncated:
        observation, info = env.reset()
        print("observation, info", observation, info)
        # {'lives': 3, 'episode_frame_number': 0, 'frame_number': 1687}
        # todo: if you don't wanna quit early, comment out:
        # if terminated:
        #     break

env.close()

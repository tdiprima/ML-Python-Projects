"""
Creates an environment for a selected game using OpenAI's Gym toolkit, then randomly selects
actions to execute in the game for 1000 steps, while also rendering the game's visual output.
"""
import gym

game = "Pong-v4"
# game = "MountainCar-v0"
# game = "MsPacman-v4"

env = gym.make(game, render_mode="human")
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

env.close()

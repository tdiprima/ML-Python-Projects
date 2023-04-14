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

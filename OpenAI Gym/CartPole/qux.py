# CartPole
import gym

# env3 = gym.make("CartPole-v0") # nope
env3 = gym.make("CartPole-v1", render_mode='rgb_array')
observation = env3.reset()
# print("\nObservation:", observation)

# screen = env3.render(mode='rgb_array')  # nope
screen = env3.render()
# print("\nScreen:", screen)

print(screen)

if screen is not None:
    print(screen.shape)  # should output (400, 600, 3)

env3.close()

# CartPole
import gym

env1 = gym.make('CartPole-v1', render_mode="human")
# env1 = gym.make('CartPole-v1')
env1.reset()

for i in range(1000):
    env1.render()

    observation, reward, done, truncated, info = env1.step(env1.action_space.sample())
    if done:
        break

env1.close()

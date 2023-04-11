# CartPole
# Compare to Lunar Lander
import gym

env = gym.make('CartPole-v1', render_mode="human")
env.reset()

for i in range(1000):
    env.render()

    observation, reward, done, truncated, info = env.step(env.action_space.sample())
    if done:
        break

env.close()

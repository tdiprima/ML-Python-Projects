import gym
import pygame

env = gym.make('CartPole-v1', render_mode='human')
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _, info = env.step(action)

    if done:
        print("Episode finished")
        try:
            env.close()
            pygame.quit()
        except Exception as ex:
            print("\nI don't know what to tell ya, kid...")
            print(ex)
            exit(1)

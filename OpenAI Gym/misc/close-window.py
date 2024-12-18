"""
See: surface-quit.md
NOTE!  The problem only occurs when using jupyter notebook
"""
import gym
import pygame

env = gym.make('CartPole-v1', render_mode='human')
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)

    if done:
        print("Episode finished")
        try:
            env.close()
            pygame.quit()
        except Exception as ex:
            print("\nI don't know what to tell ya, kid...")
            print(ex)
            exit(1)


def zyzzy():
    """
    Close pygame window
    A video thing has to be running first.
    This worked before.  TODO: Read, Fix.
    https://stackoverflow.com/questions/19882415/closing-pygame-window
    """
    import pygame
    # import time

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.display.quit()
    pygame.quit()
    exit()

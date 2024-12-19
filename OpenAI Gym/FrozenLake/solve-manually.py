"""
Contains two functions - `play_frozen()` and `play_cliffwalker()` which each simulate a specific agent's journey in the
'FrozenLake-v1' and 'CliffWalking-v0' environments from OpenAI's Gym respectively, execute certain actions in each
environment, print out the new state, reward, completion status and info after each step, reset the environment and then
close it after a delay.
"""
import time

import gym


def play_frozen():
    """
    https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
    """
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
    env.reset()
    env.render()

    for x in range(2):
        # Right
        new_state, reward, done, c, info = env.step(2)
        print(new_state, reward, done, info)
        env.render()

    for x in range(3):
        # Down
        new_state, reward, done, c, info = env.step(1)
        print(new_state, reward, done, info)
        env.render()

    # Right
    new_state, reward, done, c, info = env.step(2)
    env.render()

    print('===========================================')
    print(f'Reward = {reward}')
    print(new_state, reward, done, info)

    env.reset()

    time.sleep(5)

    env.close()


def play_cliffwalker():
    """
    https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py
    """
    env = gym.make("CliffWalking-v0", render_mode="human")
    env.reset()
    env.render()

    # Up
    env.step(0)
    env.render()

    for x in range(11):
        # Right
        new_state, reward, done, c, info = env.step(1)
        print(new_state, reward, done, info)
        env.render()

    # Down
    new_state, reward, done, c, info = env.step(2)
    env.render()

    print('===========================================')
    print(f'Reward = {reward}')
    print(new_state, reward, done, info)

    env.reset()

    time.sleep(5)

    env.close()


if __name__ == '__main__':
    # play_frozen()
    play_cliffwalker()

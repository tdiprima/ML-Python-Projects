import time

import gym


def play_frozen():
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
    env.reset()
    env.render()

    for x in range(2):
        # Implement this action and move our agent in the desired direction
        new_state, reward, done, c, info = env.step(2)
        # Print out what's happening
        print(new_state, reward, done, info)
        # Display the current map to see what we're doing
        env.render()

    for x in range(3):
        new_state, reward, done, c, info = env.step(1)
        print(new_state, reward, done, info)
        env.render()

    new_state, reward, done, c, info = env.step(2)
    env.render()

    print('===========================================')
    print(f'Reward = {reward}')
    print(new_state, reward, done, info)

    env.reset()

    # Wait 5 seconds
    time.sleep(5)

    # Close the window
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

    # Wait 5 seconds
    time.sleep(5)

    # Close the window
    env.close()


if __name__ == '__main__':
    # play_frozen()
    play_cliffwalker()

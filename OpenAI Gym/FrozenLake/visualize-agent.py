"""
Visualize the agent moving on the map
"""
import sys
import time

import gym
import numpy as np
from IPython.display import clear_output

qtable = [[0.0, 0.0, 0.59049, 0.0],
          [0.0, 0.0, 0.6561, 0.0],
          [0.0, 0.729, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.81, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0455625, 0.0],
          [0.0, 0.0, 0.3290625, 0.0],
          [0.0, 0.9, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 0.0]]

environment1 = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
state = environment1.reset()[0]
done = False
sequence = []

try:
    while not done:
        # Choose the action with the highest value in the current state
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])

        # If there's no best action (only zeros), take a random one
        else:
            action = environment1.action_space.sample()

        # Add the action to the sequence
        sequence.append(action)

        # Implement this action and move the agent in the desired direction
        new_state, reward, done, _, info = environment1.step(action)

        # Update our current state
        state = new_state

    # Update the render
    clear_output(wait=True)
    environment1.render()
    time.sleep(1)  # Close the window
    environment1.close()

    print(f"\nAction sequence = {sequence}")
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print("\nType", exc_type)
    print("\nErr:", exc_obj)
    print("\nLine:", exc_tb.tb_lineno)
    sys.exit(1)

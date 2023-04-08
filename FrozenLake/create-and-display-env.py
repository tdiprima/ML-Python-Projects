"""
Make a FrozenLake-v1 environment with the gym module, display
the environment, and then close the window in 15 seconds.
"""
import time

import gym

name = 'FrozenLake-v1'
# name = 'LunarLander-v2'
# name = 'CartPole-v1'

# Create the FrozenLake-v1 environment
env = gym.make(name, render_mode="human")
env.reset()

# Display the environment
env.render()

# Wait for 15 seconds
time.sleep(15)

# Close the window
env.close()

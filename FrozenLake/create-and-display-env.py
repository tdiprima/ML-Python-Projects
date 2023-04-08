"""
Make a FrozenLake-v1 environment with the gym module, display
the environment, and then close the window in 15 seconds.
"""
import time

import gym

# Create the FrozenLake-v1 environment
env = gym.make('FrozenLake-v1', render_mode="human")
env.reset()

# Display the environment
env.render()

# Wait for 15 seconds
time.sleep(15)

# Close the window
env.close()

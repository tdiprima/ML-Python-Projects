"""
Just create an environment and display it. Close after 5 seconds.
"""
import time

import gym

# name = "FrozenLake-v1"
# name = "LunarLander-v2"
# name = "CartPole-v1"
name = "CliffWalking-v0"

# Create the FrozenLake-v1 environment
env = gym.make(name, render_mode="human")
env.reset()

# Display the environment
env.render()

# Wait for 5 seconds
time.sleep(5)

# Close the window
env.close()

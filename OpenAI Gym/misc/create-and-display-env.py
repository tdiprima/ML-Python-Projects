"""
This script creates a gym environment for a game, renders it, waits for 5 seconds and then closes the environment.
"""
import time

import gym

# game = "FrozenLake-v1"
# game = "LunarLander-v2"
# game = "CartPole-v1"
game = "CliffWalking-v0"

# Create the FrozenLake-v1 environment
env = gym.make(game, render_mode="human")
env.reset()

# Display the environment
env.render()

# Wait for 5 seconds
time.sleep(5)

# Close the window
env.close()

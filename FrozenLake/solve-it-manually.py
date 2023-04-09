import time

import gym

environment1 = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
environment1.reset()
environment1.render()

for x in range(2):
    # Implement this action and move our agent in the desired direction
    environment1.step(2)
    # Display the current map to see what we're doing
    environment1.render()

for x in range(3):
    environment1.step(1)
    environment1.render()

new_state, reward, done, c, info = environment1.step(2)
environment1.render()

print(f'Reward = {reward}')

print(new_state, reward, done, c, info)

# Reward = 1.0
# 15 1.0 True False {'prob': 1.0}

environment1.reset()
# (0, {'prob': 1})

# Wait for 5 seconds
time.sleep(5)

# Close the window
environment1.close()

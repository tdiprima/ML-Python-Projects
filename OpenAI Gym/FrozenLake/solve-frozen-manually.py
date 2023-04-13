import time

import gym

environment1 = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
environment1.reset()
environment1.render()

for x in range(2):
    # Implement this action and move our agent in the desired direction
    new_state, reward, done, c, info = environment1.step(2)
    # Print out what's happening
    print(new_state, reward, done, info)
    # Display the current map to see what we're doing
    environment1.render()

for x in range(3):
    new_state, reward, done, c, info = environment1.step(1)
    print(new_state, reward, done, info)
    environment1.render()

new_state, reward, done, c, info = environment1.step(2)
environment1.render()

print('===========================================')
print(f'Reward = {reward}')
print(new_state, reward, done, info)

environment1.reset()

# Wait 5 seconds
time.sleep(5)

# Close the window
environment1.close()

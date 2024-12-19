"""
q-learning seeks to learn a policy that maximizes the total reward
Implementing a Q-learning algorithm to train an agent to play the 'FrozenLake-v1' game in OpenAI's Gym, adjusting its
epsilon-greedy strategy over time while keeping track of rewards and displaying the training progress graphically.
"""
# import random

import gym
import numpy as np

RENDER = False  # if you want to see training set to true

if RENDER:
    env = gym.make('FrozenLake-v1', render_mode="human")
else:
    env = gym.make('FrozenLake-v1', render_mode="rgb_array")

STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS))

EPISODES = 1500  # how many times we play
MAX_STEPS = 100  # limit the number of steps, so we don't infinitely run in circles

LEARNING_RATE = 0.81  # trust the new information a lot and update our Q-value by a large amount
GAMMA = 0.96  # care about long-term rewards as well as immediate rewards

epsilon = 0.9  # prioritize exploration (but it exploits instead)
# Unless it's zero, it always seems to exploit more than explore.

rewards = []

explore = 0
exploit = 0
for episode in range(EPISODES):

    state, p = env.reset()
    for _ in range(MAX_STEPS):

        if RENDER:
            env.render()

        # Regular random and np.random
        # if random.uniform(0, 1) < epsilon:
        if np.random.uniform(0, 1) < epsilon:
            """
            Explore: select a random action
            """
            action = env.action_space.sample()
            explore += 1
        else:
            """
            Exploit: select the action with max value (future reward)
            """
            action = np.argmax(Q[state, :])
            exploit += 1

        next_state, reward, done, truncated, info = env.step(action)

        # Update q values
        # Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

        # if np.max(Q[next_state, :]) > 0.0:
        #     # Break, if you're stepping through
        #     print(np.max(Q[next_state, :]))

        Q[state, action] = Q[state, action] + LEARNING_RATE * (
                reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

        # If we reached our goal (or fell in a hole)
        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break  # for now

print(f"\nExplore: {explore}, Exploit: {exploit}")
print(f"\nAverage reward: {sum(rewards) / len(rewards)}:")
# and now we can see our Q values!
print("\nQ-Table:\n", Q)

# we can plot the training progress and see how the agent improved
import matplotlib.pyplot as plt


def get_average(values):
    return sum(values) / len(values)


avg_rewards = []
for i in range(0, len(rewards), 100):
    avg_rewards.append(get_average(rewards[i:i + 100]))

plt.plot(avg_rewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()

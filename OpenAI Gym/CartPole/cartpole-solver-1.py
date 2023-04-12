import sys

import gym
import numpy as np

# Define the Q-learning hyperparameters
alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
epsilon = 0.1  # exploration rate
num_episodes = 10000  # number of episodes to run
max_steps = 500  # maximum number of steps per episode

# Create the CartPole environment
env = gym.make('CartPole-v1')

# todo: Initialize the Q-table with zeros
# num_states = (1,) * env.observation_space.shape[0]
# num_actions = env.action_space.n
# Q = np.zeros(num_states + (num_actions,))
# TRY AGAIN
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

Q = np.zeros((num_states, num_actions))

try:
    # TODO: This is wrong: np.argmax(Q[state])
    # Loop over episodes
    for episode in range(num_episodes):
        # Reset the environment and get the initial state
        state, _ = env.reset()

        # Loop over steps
        for step in range(max_steps):
            # Choose an action using epsilon-greedy policy
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                # todo: choose an action
                # IndexError: arrays used as indices must be of integer (or boolean) type
                # action = np.argmax(Q[state])
                # TRY AGAIN
                state_index = np.argmax(Q[state])
                action = state_index

            # Take the chosen action and get the next state and reward
            next_state, reward, done, truncated, info = env.step(action)

            # todo: Update the Q-value using Q-learning formula
            # Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            # TRY AGAIN
            next_state_index = np.argmax(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state_index]) - Q[state][action])

            # Update the current state
            state = next_state

            # If the game is completed, end the episode
            if done:
                break

        # Decay epsilon after each episode
        epsilon *= 0.99
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print("\nType", exc_type)
    print("\nErr:", exc_obj)
    print("\nLine:", exc_tb.tb_lineno)
    sys.exit(1)

# Close the environment
env.close()

# Print the final Q-table
print(Q)

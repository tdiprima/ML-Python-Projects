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

# Initialize the Q-table with zeros
num_states = (1,) * env.observation_space.shape[0]
num_actions = env.action_space.n

Q = np.zeros(num_states + (num_actions,))

# TODO: It's gotta be this [state] thing that's effing it up.
# Loop over episodes
for episode in range(num_episodes):
    # Reset the environment and get the initial state
    state, _ = env.reset()

    try:
        # Loop over steps
        for step in range(max_steps):
            # Choose an action using epsilon-greedy policy
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                print("[state]", [state])
                print("Q.shape[:-1]", Q.shape[:-1])
                state_index = np.ravel_multi_index([state], Q.shape[:-1])
                action = np.argmax(Q[state_index])
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print("\nType", exc_type)
        print("\nErr:", exc_obj)
        print("\nLine:", exc_tb.tb_lineno)
        sys.exit(1)

        # Take the chosen action and get the next state and reward
        next_state, reward, done, truncated, info = env.step(action)

        # Update the Q-value using Q-learning formula
        next_state_index = np.ravel_multi_index([next_state], Q.shape[:-1])
        Q[state_index][action] += alpha * (reward + gamma * np.max(Q[next_state_index]) - Q[state_index][action])

        # Update the current state
        state = next_state

        # If the game is completed, end the episode
        if done:
            break

    # Decay epsilon after each episode
    epsilon *= 0.99

# Close the environment
env.close()

# Print the final Q-table
print(Q)

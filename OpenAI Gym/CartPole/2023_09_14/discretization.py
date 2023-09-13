import gym
import numpy as np

# Discretize the state space
def discretize(state, bins):
    state_indices = []
    for dimension, bin_edges in zip(state, bins):
        state_index = np.digitize(dimension, bin_edges)
        state_indices.append(state_index)
    return tuple(state_indices)

# Define the Q-learning hyperparameters
alpha = 0.1           # learning rate
gamma = 0.99          # discount factor
epsilon = 0.1         # exploration rate
num_episodes = 10000  # number of episodes to run
max_steps = 500       # maximum number of steps per episode

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Initialize the Q-table with zeros
num_bins = [
    np.linspace(-4.8, 4.8, 24),
    np.linspace(-4, 4, 24),
    np.linspace(-0.418, 0.418, 48),
    np.linspace(-4, 4, 48),
]
Q = np.zeros(tuple(len(b) + 1 for b in num_bins) + (env.action_space.n,))

# Loop over episodes
for episode in range(num_episodes):
    # Reset the environment and get the initial state
    state = discretize(env.reset()[0], num_bins)

    # Loop over steps
    for step in range(max_steps):
        # Choose an action using epsilon-greedy policy
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        # Take the chosen action and get the next state and reward
        next_state, reward, done, truncated, info = env.step(action)
        next_state = discretize(next_state, num_bins)

        # Update the Q-value (using Q-learning formula)
        Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

        # Update the current state
        state = next_state

        # If the game is completed, end the episode
        if done:
            break

    # Decay epsilon after each episode
    epsilon *= 0.99

# Close the environment
env.close()
print(Q)

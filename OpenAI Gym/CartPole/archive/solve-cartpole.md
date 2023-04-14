## Solve cartpole

To solve the CartPole game in Python, you can use a reinforcement learning algorithm such as:

1. Q-learning
1. [SARSA](https://towardsdatascience.com/reinforcement-learning-with-sarsa-a-good-alternative-to-q-learning-algorithm-bf35b209e1c)

or deep reinforcement learning algorithms such as:

1. deep Q-networks (DQN) 
1. policy gradient methods

Example implementation of a basic Q-learning algorithm for solving the CartPole game using the gym module:

```python
import gym
import numpy as np

env = gym.make('CartPole-v1')

# Set the hyperparameters
num_episodes = 5000
max_steps_per_episode = 200
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# Initialize the Q-table
state_space_size = env.observation_space.shape[0]
action_space_size = env.action_space.n
q_table = np.zeros((state_space_size, action_space_size))

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    reward_total = 0

    for step in range(max_steps_per_episode):
        # Exploration-exploitation trade-off
        exploration_rate_threshold = np.random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        # Take the action
        next_state, reward, done, info = env.step(action)

        # Update the Q-table
        q_table[state, action] = (1 - learning_rate) * q_table[state, action] + \
                                 learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]))

        # Update the state and total reward
        state = next_state
        reward_total += reward

        if done:
            break

    # Decay the exploration rate
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    # Print the total reward for the episode
    print(f"Episode: {episode}, Reward: {reward_total}")

# Test the agent's performance
state = env.reset()
done = False
reward_total = 0

while not done:
    action = np.argmax(q_table[state, :])
    state, reward, done, info = env.step(action)
    reward_total += reward
    env.render()

print(f"Test Reward: {reward_total}")
env.close()
```

This implementation uses the Q-learning algorithm to learn the optimal policy for the `CartPole` game.

The Q-table is initialized to all zeros, and the algorithm updates the table based on the **reward obtained** and the **maximum Q-value** of the **next state.**

The exploration-exploitation trade-off is implemented using an epsilon-greedy strategy, where the agent chooses the action with the **highest Q-value** with `probability 1 - exploration_rate`, and a **random action** with probability `exploration_rate`.

The exploration rate is decayed over time to gradually shift the agent from exploration to exploitation.

After training the agent, the code **tests the agent's performance** by running the learned policy on the environment and printing the **total reward obtained.**

## What's a policy?

A policy is like a rulebook for a video game character. It tells the character what to do in different situations to get the highest score or win the game. In the `CartPole` game, the policy tells the agent whether it should move the cart to the left or right to keep the pole balanced on top of it. The agent learns this rulebook by trying different actions and seeing which ones get it closer to winning the game.

# [Playing Pong using Reinforcement Learning](https://towardsdatascience.com/intro-to-reinforcement-learning-pong-92a94aa0f84d)

In RL land, a policy is a rule, strategy, or behavior function, that evaluates and recommends the next action given a specific state; effectively it can be thought of as **a map from state to action.**

A policy may be deterministic or stochastic in nature, and since the ultimate goal of a RL agent is to maximize its rewards, we want to select the policy that maximizes the future expected reward for a given action.

The policy in the case of Pong helps the Agent select one of the possible actions, which are moving the paddle up, down or doing nothing.

## Discretization

It fails on the first iteration.

This line: `q_table[state, action] = (1 - learning_rate) * q_table[state, action] + learning_rate * (` fails.

I think because `state [ 0.03861446 -0.00959605  0.00653911 -0.01017574]`, but it should be 1 number.

It seems like the `state` variable is a numpy array containing **the four state variables** of the CartPole game, which are the position and velocity of the cart, and the angle and angular velocity of the pole.

The `q_table` is a 2D numpy array containing the Q-values for each state-action pair.

To update the Q-value for a given state-action pair, you need to convert the state array into an index that corresponds to the row of the `q_table` array.

This can be done using a state **discretization** function that maps the continuous state variables to a discrete set of values.

Implementation of a state discretization function that discretizes the state variables using a fixed number of bins:

```python
import numpy as np


def discretize_state(state, bins):
    state_min = [-2.4, -5, -0.209, -5]
    state_max = [2.4, 5, 0.209, 5]
    state_range = np.array(state_max) - np.array(state_min)
    state_bins = np.array(bins)

    state_index = tuple(np.clip(((state - state_min) / state_range) * state_bins, 0, state_bins - 1).astype(int))

    return state_index


# OK, HOW DO I USE IT? 😤
arr = np.array([0.03861446, -0.00959605, 0.00653911, -0.01017574])

index = discretize_state(arr, 16)
print("Index:", index)

index = discretize_state(arr, [10, 10, 10, 10])
print("Index:", index)

index = discretize_state(arr, (1, 1, 6, 3))
print("Index:", index)
```

This function takes in the `state` array and the number of bins to use for discretization, and returns a tuple of indices that correspond to the row of the `q_table` array.

The `state_min` and `state_max` variables define the minimum and maximum values for each `state` variable.

The `state_range` variable calculates the range of values for each variable.

The `state_bins` variable defines the number of bins to use for each variable.

The `state_index` variable calculates the index for each variable based on the discretization formula.

To use this function in the Q-learning algorithm, you can replace:

```ruby
q_table[state, action]
# with
q_table[discretize_state(state, bins)][action]
```

For example:

```ruby
state_index = discretize_state(state, bins)
q_table[state_index][action] = (1 - learning_rate) * q_table[state_index][action] + learning_rate * (reward + discount_factor * np.max(q_table[discretize_state(next_state, bins)]))
```

By discretizing the state variables, you can map the continuous state space to a discrete set of values that can be easily indexed in the q_table array. This allows the algorithm to learn the optimal policy even for high-dimensional state spaces.

<span style="color:#0000dd;font-size:larger;">Here we go...</span>

## What bins??

The implementation code uses a variable called "bins".  As in `state_index = discretize_state(state, bins)`.  So in solving the CartPole game... which variable is `bins`?

In the context of the `CartPole` game, the `bins` variable determines the number of discrete values to use for each state variable during the state discretization step. In other words, it defines the granularity of the discretization.

For example, if you set `bins = [10, 10, 10, 10]`, the state space will be divided into 10 bins for each of the four state variables, resulting in a total of 10^4 = 10,000 discrete states. Increasing the number of bins will increase the resolution of the state space, but will also increase the size of the Q-table, and therefore the amount of memory required to store it.

Choosing an appropriate value for `bins` requires some experimentation and depends on the complexity of the game and the available computing resources. In general, a higher number of bins will result in better performance, but at the cost of higher memory usage and longer training times. A lower number of bins may result in faster training times, but at the cost of lower performance.

In the implementation code you provided, `bins = (1, 1, 6, 3)` <span style="color:red;">(where??)</span> is used, which means that the first two state variables (position and velocity of the cart) are not discretized, while the last two state variables (angle and angular velocity of the pole) are discretized into 6 and 3 bins, respectively.

# Here.

```py
import gymimport numpy as npenv = gym.make('CartPole-v1')# state = env.reset()state, _ = env.reset()  # Extract the state and ignore the empty dictionary# Define the number of bins for each state variablenum_bins = [16, 16, 16, 16]# Define the range for each state variablestate_range = [(-4.8, 4.8), (-5.0, 5.0), (-0.42, 0.42), (-5.0, 5.0)]# Compute the bin edges for each state variablebin_edges = [np.linspace(start, stop, num + 1)[1:-1] for (start, stop), num in zip(state_range, num_bins)]# Discretize the state into a scalar indexstate_index = np.ravel_multi_index([np.digitize(s, bins) for s, bins in zip(state, bin_edges)], num_bins)# Print the state and its corresponding indexprint(f"State: {state}")print(f"State index: {state_index}")```

<br>


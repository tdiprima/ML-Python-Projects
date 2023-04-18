"""
q-learning seeks to learn a policy that maximizes the total reward
"""
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

EPISODES = 1500  # how many times to run the environment from the beginning
MAX_STEPS = 100  # max number of steps allowed for each run of environment

LEARNING_RATE = 0.81  # learning rate
GAMMA = 0.96

epsilon = 0.9

rewards = []
for episode in range(EPISODES):

    state, p = env.reset()
    for _ in range(MAX_STEPS):

        if RENDER:
            env.render()

        if np.random.uniform(0, 1) < epsilon:
            """
            Explore: select a random action
            """
            action = env.action_space.sample()
        else:
            """
            Exploit: select the action with max value (future reward)
            """
            action = np.argmax(Q[state, :])

        next_state, reward, done, truncated, info = env.step(action)

        # Update q values
        # Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

        Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break  # reached goal

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

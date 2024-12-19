"""
Creates an agent that learns to play the game 'MountainCar-v0' from the Gym environment
using Q-learning, visualizes the results, and saves the Q-table and scores.
"""
# import time
# import random
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np

# Create our environment
# env = gym.make('MountainCar-v0')
env = gym.make('MountainCar-v0', render_mode='rgb_array')
env._max_episode_steps = 1000
# env.seed(1)

# View our environment
print("The size of state is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)


def show_matplot():
    print("show_matplot")
    env.reset()
    plt.figure()
    plt.imshow(env.render())
    plt.title('Original Frame')
    plt.show()
    env.close()  # CLOSE


# show_matplot()


def random_play():
    print("play Cartpole with a random policy")
    env = gym.make('MountainCar-v0', render_mode='human')
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        state, reward, done, t, c = env.step(action)
        if done:
            env.close()  # CLOSE
            break

    env.close()  # CLOSE


# random_play()

# Creating State Preprocess Function
pos_space = np.linspace(-1.2, 0.6, 12)
vel_space = np.linspace(-0.07, 0.07, 20)


def get_state(state):
    pos, vel = state
    pos_bin = np.digitize(pos, pos_space)
    vel_bin = np.digitize(vel, vel_space)
    return pos_bin, vel_bin


# Creating out agent
GAMMA = 0.99  # discount factor
LR = .1  # learning rate


class Agent:
    def __init__(self, p_size=12, v_size=20, action_size=3):
        self.action_space = list(range(0, action_size))

        # Initializing state
        self.states = []
        for p in range(p_size + 1):
            for v in range(v_size + 1):
                self.states.append((p, v))

        # Initializing Q
        self.Q = {}
        for s in self.states:
            for a in self.action_space:
                self.Q[s, a] = 0

    def act(self, state, eps=.0):
        """Returns actions for given state as per action value function."""
        state = get_state(state)
        values = np.array([self.Q[state, a] for a in self.action_space])
        action = np.argmax(values)
        g_action = np.random.choice(self.action_space) if np.random.random() < eps else action
        return g_action

    def step(self, s, a, ns, r):
        na = self.act(ns)
        s = get_state(s)
        ns = get_state(ns)
        self.Q[s, a] = self.Q[s, a] + LR * (r + (GAMMA * self.Q[ns, na]) - self.Q[s, a])


# Watching untrained agent play
agent = Agent(p_size=12, v_size=20, action_size=3)


def ladadeeladeda():
    print("watch an untrained agent")
    env = gym.make('MountainCar-v0', render_mode='human')
    state = env.reset()[0]
    for j in range(200):
        action = agent.act(state, 0.5)
        env.render()
        next_state, reward, done, t, c = env.step(action)
        state = next_state
        if done:
            break

    env.close()  # CLOSE


# ladadeeladeda()

# env = gym.make('MountainCar-v0', render_mode='human')
env = gym.make('MountainCar-v0', render_mode='rgb_array')  # faster, please

# Loading Agent
start_epoch = 0
eps_start = 1.0
scores_window = deque(maxlen=100)  # DEQUE!


# To Load checkpoint uncomment code
# checkpoint = np.load('mountain_climb_qlearning.npy', allow_pickle=True)
# agent.Q = checkpoint[()]['Q']
# scores = checkpoint[()]['scores']
# eps_start = checkpoint[()]['eps_start']
# start_epoch = len(scores)

# index = 1
# for i in reversed(scores):
#     scores_window.append(i)
#     if index == 100:
#         break
#     index += 1

# Train the Agent with DQN
def train(n_episodes=1000, eps_end=0.01):
    print("Train the Agent with DQN")
    global eps_start
    scores = []
    eps = eps_start

    for i_episode in range(start_epoch + 1, n_episodes + 1):
        state = env.reset()[0]
        score = 0

        while True:
            action = agent.act(state, eps)
            next_state, reward, done, t, c = env.step(action)
            agent.step(state, action, next_state, reward)
            score += reward
            state = next_state

            if done:
                break

        eps = eps - 2 / n_episodes if eps > eps_end else eps_end  # decrease epsilon
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps_start = eps

        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilion: {:.2f}'.format(i_episode, np.mean(scores_window), eps),
              end="")
        if i_episode % 1000 == 0:
            print(
                '\rEpisode {}\tAverage Score: {:.2f}\tEpsilion: {:.2f}'.format(i_episode, np.mean(scores_window), eps))

    return scores


# LET'S GO!
skores = train(10)
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(skores)), skores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

data = {
    "Q": agent.Q,
    "scores": skores,
    "eps_start": eps_start
}
np.save("mountain_climb_qlearning.npy", data)

print("Done!")

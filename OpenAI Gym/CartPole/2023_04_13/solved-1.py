"""
This script trains a model for the CartPole-v1 game on OpenAI Gym using Q-learning, with an epsilon-decay strategy
for exploration-exploitation balance, and outputs performance metrics for each run in a graph.

https://github.com/JackFurby/CartPole-v0/blob/master/cartPole.py
"""
import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('CartPole-v1')

# How much new info will override old info. 0 means nothing is learned, 1 means only most recent is considered, old knowledge is discarded
LEARNING_RATE = 0.1
# Between 0 and 1, measure of how much we care about future reward over immediate reward
DISCOUNT = 0.95
RUNS = 10000  # Number of iterations run
SHOW_EVERY = 2000  # How often the current solution is rendered
UPDATE_EVERY = 100  # How often the current progress is recorded

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = RUNS // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


def create_bins_and_q_table():
    """
    Create bins and Q table
    :return:
    high: [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
    low: [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]
    obs_space_size: 4
    """
    # todo: Remove hard coded values when I know how to
    num_bins = 20
    obs_space_size = len(env.observation_space.high)

    # Get the size of each bucket
    m_bins = [
        np.linspace(-4.8, 4.8, num_bins),
        np.linspace(-4, 4, num_bins),
        np.linspace(-.418, .418, num_bins),
        np.linspace(-4, 4, num_bins)
    ]

    q_table = np.random.uniform(low=-2, high=0, size=([num_bins] * obs_space_size + [env.action_space.n]))

    return m_bins, obs_space_size, q_table


def get_discrete_state(state, m_bins, obs_space_size):
    """
    Given a state of the environment, return its discreteState index in qTable
    :param state:
    :param m_bins:
    :param obs_space_size:
    :return:
    """
    state_index = []
    for i in range(obs_space_size):
        state_index.append(np.digitize(state[i], m_bins[i]) - 1)  # -1 will turn bin into index

    return tuple(state_index)


# AND... GO!
bins, obsSpaceSize, qTable = create_bins_and_q_table()

# print('===========================================')
# print('Q-table before training:')
# print(qTable)  # not zeroes
print("\nQ-table len, shape:", len(qTable), qTable.shape)

previousCnt = []  # array of all scores over runs
metrics = {'ep': [], 'avg': [], 'min': [], 'max': []}  # metrics recorded for graph

for run in range(RUNS):
    my_state, xxx = env.reset()
    discreteState = get_discrete_state(my_state, bins, obsSpaceSize)
    done = False  # has the environment finished?
    cnt = 0  # how many movements cart has made

    while not done:
        # if run % SHOW_EVERY == 0:
        #     env.render()  # if running RL comment this out

        cnt += 1
        # Get action from Q table
        if np.random.random() > epsilon:
            action = np.argmax(qTable[discreteState])
        # Get random action
        else:
            action = np.random.randint(0, env.action_space.n)
        newState, reward, done, t, info = env.step(action)  # perform action on enviroment

        newDiscreteState = get_discrete_state(newState, bins, obsSpaceSize)

        maxFutureQ = np.max(qTable[newDiscreteState])  # estimate of optimal future value
        currentQ = qTable[discreteState + (action,)]  # old value

        # pole fell over / went out of bounds, negative reward
        if done and cnt < 200:
            reward = -375

        # formula to calculate all Q values
        newQ = (1 - LEARNING_RATE) * currentQ + LEARNING_RATE * (reward + DISCOUNT * maxFutureQ)
        qTable[discreteState + (action,)] = newQ  # Update qTable with new Q value

        discreteState = newDiscreteState

    previousCnt.append(cnt)

    # Decaying is being done every run if run number is within decaying range
    if END_EPSILON_DECAYING >= run >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # Add new metrics for graph
    if run % UPDATE_EVERY == 0:
        latestRuns = previousCnt[-UPDATE_EVERY:]
        averageCnt = sum(latestRuns) / len(latestRuns)
        metrics['ep'].append(run)
        metrics['avg'].append(averageCnt)
        metrics['min'].append(min(latestRuns))
        metrics['max'].append(max(latestRuns))
        print("Run:", run, "Average:", averageCnt, "Min:", min(latestRuns), "Max:", max(latestRuns))

env.close()

# Plot graph
plt.plot(metrics['ep'], metrics['avg'], label="average rewards")
plt.plot(metrics['ep'], metrics['min'], label="min rewards")
plt.plot(metrics['ep'], metrics['max'], label="max rewards")
plt.legend(loc=4)
plt.show()

import sys

try:
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
        state = env.reset()  # (array([ 0.04098918,  0.02709236,  0.02586222, -0.02846243], dtype=float32), {})
        state = env.action_space.sample()
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
            next_state, reward, done, _, info = env.step(action)
            # [ 0.0138943   0.19428122 -0.01284254 -0.27444175] 1.0 False False {}

            decay_factor = 1 - learning_rate
            old_value = decay_factor * q_table[state, action] + learning_rate
            max_val = np.max(q_table[next_state, :])
            new_value = (reward + discount_factor * max_val)

            # Update the Q-table
            q_table[state, action] = old_value * new_value

            # Update the state and total reward
            state = next_state
            reward_total += reward

            if done:
                break

        linear_interpolation = min_exploration_rate + (max_exploration_rate - min_exploration_rate)
        exponential_decay = np.exp(-exploration_decay_rate * episode)

        # Decay the exploration rate
        exploration_rate = linear_interpolation * exponential_decay

        # Print the total reward for the episode
        print(f"Episode: {episode}, Reward: {reward_total}")

    # Test the agent's performance
    state = env.reset()
    done = False
    reward_total = 0

    while not done:
        action = np.argmax(q_table[state, :])
        state, reward, done, _, info = env.step(action)
        reward_total += reward
        env.render()

    print(f"Test Reward: {reward_total}")
    env.close()

except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print("\nType", exc_type)
    print("\nErr:", exc_obj)
    print("\nLine:", exc_tb.tb_lineno)
    sys.exit(1)

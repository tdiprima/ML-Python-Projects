import gym


def choose_action(state):
    angle, _, pole_velocity, _ = state
    if angle < 0:
        return 0  # move left
    else:
        return 1  # move right


# Create the CartPole environment
env = gym.make('CartPole-v1')

# Reset the environment and get the initial state
state, _ = env.reset()

# Run the game until completion
done = False
epochs = 0
while not done:
    # Choose an action based on the current state
    action = choose_action(state)  # represents your algorithm

    # Take the chosen action and get the next state and reward
    next_state, reward, done, truncated, info = env.step(action)

    # Update the current state
    state = next_state

    epochs += 1

print("Epochs:", epochs)

# Close the environment
env.close()

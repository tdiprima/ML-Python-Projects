"""
This script runs some random games in the CartPole-v1 environment of OpenAI's gym, where for each frame in each game,
it samples an action (either 0 or 1, which corresponds to the pole moving left or right), executes the action in the
environment, and prints the reward if the game is done.

https://github.com/openai/gym/issues/762
https://pythonprogramming.net/static/downloads/ipynbs/Reinforcement+Learning+-+Playing+Games+with+Neural+Networks+and+Artificial+Intelligence.html
"""
import gym

LR = 1e-3
env = gym.make("CartPole-v1", render_mode="human")

env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000


def some_random_games_first():
    # Each of these is its own game.
    for episode in range(5):
        env.reset()
        # this is each frame, up to 200... but we won't make it that far.
        for t in range(200):
            # This will display the environment
            # todo: Only display if you really want to see it.
            # Takes much longer to display it.
            # env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, done, truncated, info = env.step(action)

            if done:
                print("reward:", reward)
                break


some_random_games_first()

env.close()

## What is the object of the game CartPole?

Balance.

The object of the game `CartPole` in Python's gym module is to balance a pole on top of a cart that moves along a frictionless track. The game is a classic control problem in reinforcement learning, where the agent (in this case, the program that you write) must learn to balance the pole for as long as possible by controlling the movements of the cart.

The game has a state space of **four dimensions:** the position and velocity of the cart, and the angle and angular velocity of the pole.

The action space consists of **two possible actions**: move the cart to the left or to the right.

The game is considered "solved" if the agent can balance the pole for 200 time steps (or frames) in a row. However, the ultimate goal is to balance the pole for as long as possible, ideally indefinitely.

`CartPole` is a popular benchmark problem for **testing reinforcement learning algorithms,** as it is simple yet challenging enough to require the agent to learn a good policy for balancing the pole.

## Changes after testing the gym's cartPole error

Code block seen on the Internet:

```py
import gym  # Import the Python interface environment package for Gym

env = gym.make('CartPole-v0')  # Build the experimental environment
env.reset()  # reset a round

for _ in range(1000):
    env.render()  # show GUI
    action = env.action_space.sample() # Randomly pick an action from the action space
    env.step(action) # Used to submit actions, the specific actions are in brackets

env.close() # close environment
```

There will be 3 errors, the modified code:

```py
env = gym.make("CartPole-v1", render_mode="human")

for episode in range(10):
    env.reset()
    print("Episode finished after {} timesteps".format(episode))
    for ik in range(100):
        env.render()
        observation, reward, done, info, _ = env.step(env.action_space.sample())
        if done:
            break
        time.sleep(0.02)
env.close()
```

**First error:**  UserWarning: WARN: The environment `CartPole-v0` is **out of date.** You should consider upgrading to version `v1`.

Solution:  CartPole **-v0** changed to: CartPole **-v1**

**Second error:** UserWarning: WARN: You are calling render method without specifying any render mode.

Solution:

```py
env = gym.make('CartPole-v1', render_mode="human")
```

**Third error:** UserWarning: WARN: You are calling `step()` even though **this environment has already returned `terminated = True`**

You should **always call `reset()`** once you receive `terminated = True`.  Any further steps are undefined behavior.

**Solution:**

When `done` is equal to `true`, it is still in `step`.

Just add a **judgment condition**, change it to:

```py
observation, reward, done, info, _ = env.step(env.action_space.sample())
if done:
    break
```

[Article](https://blog.csdn.net/insid1out/article/details/128654843)

[Interesting: taxi.py](https://raw.githubusercontent.com/openai/gym/master/gym/envs/toy_text/taxi.py)

<br>

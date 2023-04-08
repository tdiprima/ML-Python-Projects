## Changes after testing the gym's cartPole error

<span style="color: green; font-weight:bold; font-size: larger">Reinforcement_Learning.ipynb</span>

<span style="color: blue; font-weight:bold; font-size: larger">FrozenLake-v0</span>

[Article](https://blog.csdn.net/insid1out/article/details/128654843)

[Interesting: taxi.py](https://raw.githubusercontent.com/openai/gym/master/gym/envs/toy_text/taxi.py)

### Code block seen on the Internet:

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

### There will be 3 errors, the modified code:

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

## First error:

UserWarning: WARN: The environment `CartPole-v0` is **out of date.** You should consider upgrading to version `v1`.

### Solution:

CartPole **-v0** changed to: CartPole **-v1**

## The second error:

UserWarning: WARN: You are calling render method without specifying any render mode.

You can **specify the `render_mode` at initialization**, e.g.

```py
gym("CartPole-v0", render_mode="rgb_array")
```

### Solution:

This sentence:

```py
env = gym.make('CartPole-v0') # Build the experimental environment
```

<br>
Change to:

```py
env = gym.make('CartPole-v1', render_mode="human")
```

## Third error:

UserWarning: WARN: You are calling `step()` even though **this environment has already returned `terminated = True`**

You should **always call `reset()`** once you receive `terminated = True` &ndash; any further steps are undefined behavior.

### Solution:

When `done` is equal to `true`, it is still in `step`.

Just add a **judgment condition**, change it to:

```py
observation, reward, done, info, _ = env.step(env.action_space.sample())
if done:
    break
```

<br>

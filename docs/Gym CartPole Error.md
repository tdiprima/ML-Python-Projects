## Changes after testing the gym's cartPole error

<!-- https://blog.csdn.net/insid1out/article/details/128654843 -->

<!--Interesting: https://raw.githubusercontent.com/openai/gym/master/gym/envs/toy_text/taxi.py-->

Go back today and re-run the gym's CartPole use case. After updating the version of the package, an error occurred.

python version: Python 3.9.13

gym version: gym 0.26.2

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

There will be three errors, the modified code

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

### First error:

UserWarning: WARN: The environment CartPole-v0 is out of date. You should consider upgrading to version `v1`.

### Solution:

'CartPole-v0' changed to: 'CartPole-v1'

###The second error:

UserWarning: WARN: You are calling render method without specifying any render mode. You can specify the render\_mode at initialization, eg gym("CartPole-v0", render\_mode="rgb_array") 

### Solution:

This sentence:

```py
env = gym.make('CartPole-v0') # Build the experimental environment
```

Change to:

```py
env = gym.make('CartPole-v1', render_mode="human")
```

### Third error:

UserWarning: WARN: You are calling 'step()' even though this environment has already returned terminated = True. You should always call 'reset()' once you receive 'terminated = True' &ndash; any further steps are undefined behavior.

When done is equal to true, it is still in step, just add a judgment condition and
change it to

```py
observation, reward, done, info, _ = env.step(env.action_space.sample())
if done:
    break
```

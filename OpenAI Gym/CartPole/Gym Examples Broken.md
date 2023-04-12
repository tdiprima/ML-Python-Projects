## Buckle up.

First of all:

**Make environment, reset environment, render environment.**

### None of the examples work?

Check the version of the `gym` library you're using and compare it to the version used in the examples you found.

### Nothing renders?

When I do `environment.render()`, nothing renders.

You need to put a `render_mode`.  Specifically, `render_mode=human`.

And set the environment first, obviously.

### Results of env.step(action)?

Everyone says 4 values are returned, should be 5.

~~observation, reward, done, info = env.step(action)~~

```ruby
observation, reward, done, truncated, info = env.step(action)

next_state, reward, terminated, truncated, info = env.step(action)
```

<br>
See: [OpenAI GYM's env.step(): what are the values?](https://stackoverflow.com/questions/73195438/openai-gyms-env-step-what-are-the-values)

### Module object not callable

The error message tells you to do something, but it's wrong 

```py
# env = gym("CartPole-v0")  # don't do this
env = gym.make("CartPole-v0")  # do this
```

The error is: **'module' object is not callable.**

Meaning: you're trying to call a module as if it were a function.

### Don't fall for it

Don't fall for it if someone tells you to do this:

```python
rendered = env.render(mode="rgb_array")
```

First of all, "mode" is wrong; it should be `render_mode`.

Second, `render_mode` doesn't go there!  It goes in when you create the environment.


### Rendering as an RGB array

I wasn't trying to get an rgb array!

To get the RGB array, you can instead use the `get_screen()` method, which returns a 400x600 RGB image of the environment's current screen. Here's an example code snippet:

```py
import gym

env = gym.make("CartPole-v0", render_mode='rgb_array')
observation = env.reset()
screen = env.render()

print(screen.shape)  # (400, 600, 3)
env.close()
```

### After quux and corge

Nothing we are doing is working.  When I do a Google search, all that comes up is code that fails.  There's got to be a working piece of code somewhere.

Buddy then proceeds to write code that again, does not work.  See garply.py

## Reset

To set the state of an environment in Gym, you should use the `env.reset()` method.

The `reset()` method resets the environment to its initial state and returns the initial observation. You can then take actions in the environment to change its state.

To set a specific state in an environment, you can take the following steps:

* Call the `reset()` method to set the environment to its initial state and get the initial observation.
* Perform a sequence of actions in the environment to reach the desired state. For example, you could randomly sample actions and apply them to the environment until you reach the desired state.
* Once you reach the desired state, you can return the observation obtained from the last action as the new state.
* Alternatively, some environments in Gym may provide a custom `set_state()` method to set the state to a specific value. You should check the documentation of the environment you're using to see if such a method is available and how to use it.

[Environment working, but not render](https://github.com/openai/gym/issues/762)

First example doesn't solve it.

[Import error when making cliff-v0 environment](https://github.com/openai/gym/issues/2779)

<br>

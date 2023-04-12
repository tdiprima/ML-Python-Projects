## Pygame display surface

[www.pygame.org](https://www.pygame.org/docs/ref/display.html)

[gym env.step returns](https://stackoverflow.com/questions/73195438/openai-gyms-env-step-what-are-the-values)

```ruby
next_state, reward, terminated, truncated, info = env.step(action)
```

<br>
In the context of a Gym environment with `render_mode='human'`, the Pygame display surface is created when you call `env.render()` for the first time.

If the display surface is **closed** or **uninitialized** (e.g., because you called `env.close()` or because an error occurred), any subsequent attempts to access the display surface will result in the "display Surface quit" error.

To avoid this error, you should make sure that you're only calling `env.render()` when the environment is open and the display surface is available.

Here's how to avoid the "display Surface quit" error:

```python
import gym

# env = gym.make('MountainCar-v0', new_step_api=True)
env = gym.make('CartPole-v1', render_mode='human')
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)

    if done:
        print("Episode finished")
        # check if the environment is open
        # if env._open:
        if env.env is not None:
            env.close()

    # if env._open:
    if env.env is not None:
        env.render()
```

We enter a loop where we sample an action from the environment's action space and take a step in the environment.

We check if the environment is open before calling `render()` or `close()`, to avoid the "display Surface quit" error.

## Spinning pinwheel

<span style="color:#0000dd;font-size:larger;">The problem only occurs when using jupyter notebook.  The End.</span>

It looks like the window wants to close.  But I'm getting a spinning pinwheel...

If you're using the `render_mode='human'` option, it's possible that the window may become **unresponsive** or **freeze**, especially if you're running a **long episode.**

In this case, you can use a try-except block to catch any exceptions that might occur when you try to close the window, and force-close it using the pygame module.

Note that this may not work if the window is completely unresponsive, in which case you may need to **force-quit** the Python script.

```python
import gym
import pygame

env = gym.make('CartPole-v1', render_mode='human')
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _, info = env.step(action)
    
    # Close the window when done/terminated=True
    if done:
        print("Episode finished")
        try:
            env.close()
        except:
            pygame.quit()
```

<br>

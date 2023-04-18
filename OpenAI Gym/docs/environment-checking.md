## Gym Environment Checker

Gym is **reinforcement learning** training.

```py
from gym.utils.env_checker import check_env

check_env(environment)
```

I got an error message that says the environment has a **wrapper applied to it**, which can potentially affect the environment checker's performance.

The recommended solution is to use the **raw environment** for `check_env` using `env.unwrapped`.  

How?

**`unwrapped`** returns the underlying raw environment with all the **wrappers removed.**

```python
import gym
from gym.utils import env_checker

env = gym.make("CartPole-v1", render_mode="human")
raw_env = env.unwrapped
result = env_checker.check_env(raw_env)

print(f"\nType: {type(result)}\nThing: {result}")

env.close()
```

By using the unwrapped method, you are ensuring that the environment checker is applied to the raw environment without any wrappers, providing a **more accurate evaluation** of the environment's behavior.

## Env checker output summary

When using `gym.utils` `env_checker`, what should the output be?  How do I know if the environment is right or not?

The output should be a **string** that describes any issues found in the environment.

If no issues are found, the output will be an empty string.  <span style="color:#0000dd;font-size:larger;">Or "None", as it were.</span>

<br>


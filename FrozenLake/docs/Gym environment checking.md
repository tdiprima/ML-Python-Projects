```commandline
I'm using "gym.utils.env_checker  import check_env".  I did `check_env(environment)`. 
Then I got an error:
"The environment (<TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>) 
is different from the unwrapped version (<CartPoleEnv<CartPole-v1>>). 
This could effect the environment checker as the environment most likely 
has a wrapper applied to it. We recommend using the raw environment for 
`check_env` using `env.unwrapped`."
```

The error message indicates that the environment you are using has a wrapper applied to it, which can potentially affect the environment checker's performance.

The recommended solution is to use the raw environment for `check_env` using `env.unwrapped`.

The unwrapped method returns the underlying raw environment with all the wrappers removed.

Here's an example of how you can use the **unwrapped** method:

```python
import gym
from gym.utils import env_checker

env = gym.make('CartPole-v1')
raw_env = env.unwrapped

env_checker.check_env(raw_env)
```

By using the unwrapped method, you are ensuring that the environment checker is applied to the raw environment without any wrappers, providing a more accurate evaluation of the environment's behavior.


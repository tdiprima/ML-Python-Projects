# TL;DR - just forget it for now.

```sh
pip install pettingzoo
pip install "pettingzoo[atari]"

# missing "pymunk", etc, so

pip install "pettingzoo[all]"

# then, error
# ERROR: pip's dependency resolver does not currently take into 
# account all the packages that are installed. This behaviour is # the source of the following dependency conflicts.
# spyder 5.3.3 requires pyqt5<5.16, which is not installed.
# spyder 5.3.3 requires pyqtwebengine<5.16, which is not installed.
```

Seems like a good article: [Using PettingZoo with RLlib for Multi-Agent Deep Reinforcement Learning](https://towardsdatascience.com/using-pettingzoo-with-rllib-for-multi-agent-deep-reinforcement-learning-5ff47c677abd[]())

It's out of date already, but looks like it might be helpful anyway.

# Render

How do I render a PettingZoo environment for 5 seconds, and then close the window?

You can use the pettingzoo.utils.Wrapper class from the PettingZoo library along with the pygame module.

```python
import pygame
from pettingzoo.utils import Wrapper

# create the PettingZoo environment and wrap it with the pygame display
env = Wrapper(env_name="environment_name", visualization=True)
env.initialize_output()

# render the environment for 5 seconds
pygame.time.set_timer(pygame.USEREVENT + 1, 5000)  # set timer for 5 seconds
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            pygame.quit()
            quit()
        elif event.type == pygame.USEREVENT + 1:
            env.close()
            pygame.quit()
            quit()
    env.render()
```

In the code snippet above, we first import the pygame module and the Wrapper class from the pettingzoo.utils module. We then create the PettingZoo environment and wrap it with the Wrapper class with the visualization flag set to True to enable rendering. We initialize the output to start the rendering process.

Next, we set a timer for 5 seconds using the pygame.time.set\_timer() method with a custom event type (pygame.USEREVENT + 1). Then we enter a loop where we continuously check for events. If the QUIT event is detected, we close the environment and quit the program. If the custom event is detected (i.e., the 5-second timer has elapsed), we also close the environment and quit the program. In each iteration of the loop, we call the render() method to update the display.

Note that you will need to replace "environment\_name" with the actual name of the PettingZoo environment you want to render. Also, make sure that you have the necessary dependencies installed, including pygame and pettingzoo.


# Error

I'm getting: "cannot import name 'Wrapper' from 'pettingzoo.utils'"

If you are getting an error message "cannot import name 'Wrapper' from 'pettingzoo.utils'" while trying to use the Wrapper class from the pettingzoo.utils module, it may be because you are using an older version of the PettingZoo library. The Wrapper class was introduced in version 1.10.0 of PettingZoo.

To fix this issue, you can try upgrading your PettingZoo installation to the latest version.

```sh
pip install -U pettingzoo
```

This will upgrade your PettingZoo installation to the latest version available on PyPI.

If you have already installed the latest version of PettingZoo, and you are still encountering this error, you can try importing the make\_env function from the pettingzoo module instead of the Wrapper class.

```python
import pygame
import pettingzoo
import numpy as np

# create the PettingZoo environment
env = pettingzoo.make_env("environment_name")
# no make_env

# initialize the environment
obs = env.reset()

# set up the pygame display
pygame.init()
screen = pygame.display.set_mode((env.render_modes["human"]["width"], env.render_modes["human"]["height"]))
# env.render_mode["human"]["height"] not subscriptable

# render the environment for 5 seconds
pygame.time.set_timer(pygame.USEREVENT + 1, 5000)  # set timer for 5 seconds
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            pygame.quit()
            quit()
        elif event.type == pygame.USEREVENT + 1:
            env.close()
            pygame.quit()
            quit()

    # get the current observation from the environment
    obs, _, _, _ = env.step(np.zeros(env.num_agents))

    # render the observation
    screen.blit(obs[0], (0, 0))
    pygame.display.update()
```

In this code snippet, we use the make\_env function from the pettingzoo module to create the PettingZoo environment. We then initialize the environment by calling the reset method, which returns the initial observation.

Next, we set up the pygame display and render the environment for 5 seconds using the same approach as in the previous code snippet. In each iteration of the loop, we call the step method to advance the environment by one time step and obtain the new observation. We then use the blit method to render the observation onto the screen and call the pygame.display.update() method to update the display.

Note that you will need to replace "environment\_name" with the actual name of the PettingZoo environment you want to render. Also, make sure that you have the necessary dependencies installed, including pygame and pettingzoo.

<span style="color:red;">This is all because I wanted to run an old version of gym, and in doing so, I discovered a shiny new toy.  And of course &ndash; it doesn't work.</span>

## Things that I wanted to play with

[Ice Hockey](https://pettingzoo.farama.org/environments/atari/ice_hockey/)

[Mario Bros](https://pettingzoo.farama.org/environments/atari/mario_bros/)

[Combat: Tank](https://pettingzoo.farama.org/environments/atari/combat_tank/)

[Combat: Plane](https://pettingzoo.farama.org/environments/atari/combat_plane/)


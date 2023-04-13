## A.L.E: Arcade Learning Environment

### [Powered by Stella]

### Install

```sh
pip install gymnasium[atari]

pip install gymnasium[accept-rom-license]
```

<!--gym.make("MsPacman-v4")
gymnasium.make("ALE/MsPacman-v5")
-->

### Basic Usage

```py
import gym
env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## There's Space Invaders, too!

["gymnasium" documentation](https://gymnasium.farama.org/)

[Atari Complete List](https://gymnasium.farama.org/environments/atari/complete_list/)

<!--[Gym Documentation](https://www.gymlibrary.dev/)-->

[MsPacman](https://gymnasium.farama.org/environments/atari/ms_pacman/)

[The Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)

<!--[Ms Pacman | Gym](https://www.gymlibrary.dev/environments/atari/ms_pacman/)-->

<!--[Atari](https://www.gymlibrary.dev/environments/atari/)-->

[How to Train Ms-Pacman with Reinforcement Learning](https://medium.com/analytics-vidhya/how-to-train-ms-pacman-with-reinforcement-learning-dea714a2365e)

[Automating Pac-man with Deep Q-learning: An Implementation in Tensorflow](https://towardsdatascience.com/automating-pac-man-with-deep-q-learning-an-implementation-in-tensorflow-ca08e9891d9c)

[Ms Pac-man AI Deep/Reinforcement Learning](https://www.youtube.com/watch?v=JM7Hf9j4v7c)

[Deep Reinforcement Learning: Playing Ms. Pac-Man](https://github.com/rooster06/Atari-Ms.PacMan)

<br>

# CartPole
# pip install moviepy
import gym
from gym.wrappers.record_video import RecordVideo

env6 = gym.make('CartPole-v1', render_mode="rgb_array")
env6 = RecordVideo(env6, './video', episode_trigger=lambda episode_number: True)

env6.reset()
env6.close()

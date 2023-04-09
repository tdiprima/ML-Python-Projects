import gym
# pip install moviepy
from gym.wrappers.record_video import RecordVideo

env = gym.make('CartPole-v1', render_mode="rgb_array")
env = RecordVideo(env, './video', episode_trigger=lambda episode_number: True)
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _, info = env.step(action)

    # Gym Environment Termination Check
    if done:
        print("Episode finished")
        env.close()

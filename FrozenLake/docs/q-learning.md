## Q-learning algorithm

<img src="https://res.cloudinary.com/dyd911kmh/image/upload/v1666973295/Q_Learning_Process_134331efc1.png" alt="datacamp.com intro Q learning">

<!--https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial-->

Q-learning is a popular algorithm used in **Reinforcement Learning**, a subfield of machine learning.

In Q-learning, we have an **"agent"** that's trying to learn how to make good decisions in a particular "environment."  For example, the agent could be a robot trying to navigate a maze, and the environment would be the maze itself.

The agent makes decisions by selecting actions. An **"action"** is just something the agent can do in the environment, like moving left or right in the maze.

The agent's goal is to find the best actions to take in each situation it encounters in the environment. We call these situations **"states."** For example, in the maze, each state would correspond to the robot's location in the maze.

The Q in Q-learning stands for **"quality."** We use a special number called the "Q-value" to represent how good an action is in a given state. So, for example, if the agent is in a particular location in the maze and it can choose to go left or right, the Q-values for those actions would represent how good they are in that particular location.

Now, the agent doesn't know the correct Q-values for each state-action pair at first. So, **it starts by making random decisions and seeing what happens.** As it gets feedback on the outcomes of its actions (in the form of rewards or penalties), it updates its Q-values to reflect what it has learned.

The **Bellman equation** is just a fancy way of saying that the Q-value for a state-action pair should be based on the immediate reward the agent receives for taking that action, plus the expected future rewards it will receive if it continues to take the best actions from that point forward.

So, the agent updates its Q-value for a state-action pair by taking the immediate reward it received, adding the maximum expected future rewards, and storing that as the new Q-value.

Over time, as the agent explores the environment and updates its Q-values, it gets better and better at making decisions that lead to good outcomes.

And that's how Q-learning works!

## Off-cannon work

Q-learning is considered an off-policy algorithm, meaning that it learns the optimal policy even if it doesn't follow it during training. This makes it particularly useful in situations where exploration is required to find the optimal policy.

<br>

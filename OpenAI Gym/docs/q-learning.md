## Q-learning algorithm

<img src="https://res.cloudinary.com/dyd911kmh/image/upload/v1666973295/Q_Learning_Process_134331efc1.png" alt="datacamp.com intro Q learning">

<a href="https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial">introduction-q-learning-beginner-tutorial</a>

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

## Off-cannon

Q-learning is considered an off-policy algorithm, meaning that it learns the optimal policy even if it doesn't follow it during training. This makes it particularly useful in situations where exploration is required to find the optimal policy.

## Q-Learning Update Rule

Q ("quality") represents how good we think each possible action is in a given situation. In other words, if we're in a certain state, what's the quality of each action we could take?

```py
# Update rule
Q[state, action] = Q[state, action] + learning_rate * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
```

<br>
Now, let's break down the different parts of the update rule:

1. **`Q[state, action]`:** This is the **current estimate** we have for the quality of taking "action" in "state". It's the Q-value for that specific state-action pair.
2. **`learning_rate`:** This is a number between 0 and 1 that determines how much we should trust new information compared to old information. If the learning rate is high, we're going to trust the new information a lot and update our Q-value by a large amount. If it's low, we're going to trust the new information less and only update our Q-value a little bit.
3. **`(reward + gamma * np.max(Q[new_state, :]) - Q[state, action])`:** This is the "error" or "residual" in our current estimate of the Q-value. It represents the difference between what we thought the Q-value should be (based on our previous experience) and what we just learned it should be (based on the new information we just received).
    * **`reward`** is the immediate reward we received for taking the action in the current state.
    * **`gamma`** is a number between 0 and 1 that represents how much we care about future rewards. If gamma is close to 0, we only care about immediate rewards. If it's close to 1, we care about long-term rewards as well.
    * **`np.max(Q[new_state, :])`** is the maximum Q-value we expect to get in the next state. In other words, it's the estimate of how good we think the best action will be in the next state.
    * **`Q[state, action]`** is our current estimate of the Q-value for the current state-action pair.

**Putting it all together:** When we update the Q-value for a state-action pair, we take the current estimate of the Q-value, add the learning rate times the error (the difference between the expected Q-value and our current estimate), and get a new estimate of the Q-value. This new estimate is hopefully closer to the true value of the Q-value, and we'll use it to make decisions in the future.

## Q[new_state, :]

Yes, `Q[new_state, :]` is an **array**. In fact, it's an array of Q-values for all the possible actions in the new state.

When we update the Q-value for a particular state-action pair, we use the Q-value for the best action in the new state. So, we want to find the maximum Q-value in the array `Q[new_state, :]` to determine the best action to take in the new state.

The **index of the maximum Q-value in the array** `Q[new_state, :]` tells us which action has the highest estimated quality in the new state. We'll use that action to update the Q-value for the current state-action pair.

[That's what Tim was talking about.  We're getting the max, not the argument max.]

So, `new_state` is indeed **the index of the row** in the Q-table that **corresponds to the new state**.

The `:` in `Q[new_state, :]` means that **we're selecting all the columns in that row,** which correspond to the Q-values for all the possible actions in the new state.

## Epsilon

What does it mean if epsilon is close to 1, and when it's close to zero?

```c
epsilon = 0.9

Explore: 3966, Exploit: 31184
```

Buddy suggests that the higher it is, the more we will explore.  But I ran it 3 times, and I got: Exploit > Explore.

Epsilon is a parameter used in reinforcement learning algorithms, such as Q-learning, to **control the trade-off** between exploration and exploitation.

Specifically, epsilon determines **the probability of choosing a random action** (exploration) versus choosing the action with the highest estimated quality (exploitation).

When epsilon is **close to 1**, it means that the algorithm will almost always choose a **random** action. This is because the probability of selecting a random action is high (close to 1), while the probability of selecting the action with the highest estimated quality is low (close to 0). In other words, the algorithm will focus on exploration, trying out different actions to see what works and what doesn't.

On the other hand, when epsilon is **close to 0**, it means that the algorithm will almost always choose the **action with the highest estimated quality.** 

This is because the probability of selecting the action with the highest estimated quality is high (close to 1), while the probability of selecting a random action is low (close to 0). In other words, the algorithm will focus on **exploitation**, always selecting the action that it thinks is the best based on its current knowledge.

So, in summary, when epsilon is close to 1, the algorithm will prioritize exploration, while when epsilon is close to 0, it will prioritize exploitation. The value of epsilon is typically decreased over time as the algorithm gains more knowledge about the environment, to gradually shift the focus from exploration to exploitation.

### I even [looked it up](https://www.baeldung.com/cs/epsilon-greedy-q-learning), but they say the same as Buddy!

Epsilon parameter is related to the epsilon-greedy action selection procedure in the Q-learning algorithm.

In the action selection step, we select the specific action based on the Q-values we already have.

The epsilon parameter **introduces randomness into the algorithm,** forcing us to try different actions. This helps not getting stuck in a local optimum.

If **epsilon is set to 0**, we never explore but always exploit the knowledge we already have.

On the contrary, having the **epsilon set to 1** force the algorithm to always take random actions and never use past knowledge.

Usually, epsilon is selected as a small number close to 0.

### Experiments

```c
epsilon = 1.0
Explore: 4642, Exploit: 27420
```

See? It doesn't "always" do the same thing; that's why we've introduced randomness!  However, it should be exploring more!!

```c
epsilon = 0
Explore: 0, Exploit: 27061
// OK, it did what it said. Never explore.
```

Try random instead of numpy random.

```py
epsilon = 0.2
Explore: 265, Exploit: 25351
# OK! That's what's supposed to happen.
```

Try high again.

```py
epsilon = 0.9
Explore: 4033, Exploit: 34255
```

Abort mission. :(

<br>

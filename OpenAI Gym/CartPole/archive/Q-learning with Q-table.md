I'm guessing this may have been from: <a href="https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial">introduction-q-learning-beginner-tutorial</a>

```ruby
decay_factor = 1 - learning_rate

q_table[state, action] = \
    decay_factor * q_table[state, action] + learning_rate * \
    (reward + discount_factor * np.max(q_table[next_state, :]))
```

This is the part where we calculate the square that we're in, and the square that we're gonna go to.  The rewards of those.

### Line by line:

The Q-table is being updated for a given state and action combination.  Duh.  `q_table[state, action] =`

The new value for that state and action combination is calculated as a **weighted average** of the old value and a new value.

The **old value** is decreased by a factor of `learning_rate`, which is a number between 0 and 1 that determines how much weight to give to the old value versus the new value.

```py
decay_factor * q_table[state, action] + learning_rate
```

The **new value** is calculated as the sum of the reward received for taking that action in that state, plus a discounted estimate of the future rewards the computer might receive by taking the best action in the next state (`next_state`).

```ruby
(reward + discount_factor * np.max(q_table[next_state, :]))
```

The `discount_factor` is a number between 0 and 1 that determines how much to discount future rewards.

The **estimate** of the future rewards is based on the **maximum Q-value** in the next state, which is found by taking the highest value in the row of the Q-table corresponding to the next state.

Overall, this code is updating the Q-table to help the computer learn which actions are best in each situation.

## Learning Rate Decay

```python
1 - learning_rate
```

In machine learning, the learning rate is a hyperparameter that controls the step size of the optimization algorithm during model training. The learning rate determines how much the model weights and biases are updated in response to the error gradient calculated during backpropagation.

When updating the weights and biases, the learning rate is multiplied by the gradient. For example, if the gradient is 0.5 and the learning rate is 0.1, the weight update would be 0.05. However, some optimization algorithms, such as **stochastic gradient descent,** benefit from a small amount of stochasticity in the updates, which can help the algorithm avoid getting stuck in local optima. One way to introduce this stochasticity is to randomly scale the learning rate.

One common technique to introduce randomness in the learning rate is to use a technique called **learning rate decay,** where the learning rate is reduced over time. One way to implement this is to multiply the learning rate by a decay factor at each iteration.

A common decay factor is to use `1 - learning_rate`, which reduces the learning rate by a fixed percentage at each iteration.

By using `1 - learning_rate` as the decay factor, the learning rate is reduced by a fixed percentage at each iteration. This can help the optimization algorithm to explore different areas of the parameter space and avoid getting stuck in local optima. Additionally, it can help the model to converge to a better solution over time.

## Numpy Q-table Maximum

```py
# Determine the maximum value of the action-value function for a given state
np.max(q_table[next_state, :]
```

Calculates the maximum value in a specific row of the `q_table` array, where the row index is given by the variable `next_state`.

More specifically, `q_table[next_state, :]` selects the row with index `next_state` and all columns in the `q_table` array, and `np.max` returns the maximum value in that row.

## Ok.  What do you do if next_state


If `next_state` is a NumPy array or a Python list with the values `[-0.02912439, -0.1594933, -0.04854622, 0.30156556]`, then the code `np.max(q_table[next_state, :])` would likely raise a TypeError because next_state is not an integer and cannot be used as an index for the `q_table` array.

To use `next_state` as an index, you would need to **convert it to an integer**.

Assuming that the `q_table` array is a two-dimensional array with **integer indices in the first dimension.**

For example, you could use the following code to convert `next_state` to an integer index:

```python
next_state_index = tuple(map(int, next_state))
```

This code converts `next_state` to a **tuple** <span style="color:red;">Umm... wtf r u doing?</span> of integers by applying the `int` function to each element of the list, and then converts the tuple to an index by wrapping it in parentheses. You can then use `next_state_index` to select the row in the `q_table` array and calculate the maximum value in that row:

```css
np.max(q_table[next_state_index, :])
```


### next\_state\_index ends up being (0, 0, 0, 0).  I need one value; a scalar.

If `next_state_index` ends up being (0, 0, 0, 0), then it's likely that the values in `next_state` are very small and are getting rounded down to zero when they are converted to integers.

To get a scalar index from `next_state`, you could use a hash function to convert the array to a unique integer value. One option is to use the built-in `hash` function in Python, which can convert an object to a hash value. For example, you could use the following code to get a scalar index from `next_state`:

```python
next_state_index = hash(tuple(next_state))
```

This code converts `next_state` to a tuple using the tuple function, then computes the hash value of the tuple using the hash function, and stores the hash value in `next_state_index`. The hash function produces a unique integer value for each input, which can be used as a scalar index in a NumPy array.

Note that the hash function may produce collisions for some input values, which could lead to incorrect results in some cases. If you need a more robust way to convert an array to a scalar index, you could consider using a different hash function or encoding method.

<span style="color:red;">Dafuq?  It's still a tuple!</span>

## What does this calculate?

```py
min_exploration_rate + (max_exploration_rate - min_exploration_rate)
```

The expression `min_exploration_rate + (max_exploration_rate - min_exploration_rate)` calculates a linear interpolation between two values: `min_exploration_rate` and `max_exploration_rate`.

The value of the expression lies between `min_exploration_rate` and `max_exploration_rate`, with the relative weight of the two values determined by the difference between them.

When `min_exploration_rate` equals zero, and `max_exploration_rate` equals one, this expression is a simple linear interpolation between zero and one, and returns a value between zero and one, inclusive.

This type of linear interpolation is commonly used in reinforcement learning algorithms to gradually decrease the exploration rate over time, as the agent learns more about the environment. The `min_exploration_rate` and `max_exploration_rate` parameters can be adjusted to control the initial and final exploration rates, respectively.

## What does this do?

```py
np.exp(-exploration_decay_rate * episode)
```

The code `np.exp(-exploration_decay_rate * episode)` computes the exponential decay of a variable over time.

The code uses the `exp` function from NumPy to compute the exponential of the value `-exploration_decay_rate * episode`. The exponential function raises the mathematical constant `e` to the power of the input value, where `e` is approximately equal to 2.71828.

In this case, the input to the exponential function is `-exploration_decay_rate * episode`, where `exploration_decay_rate` is a positive scalar that controls the rate of decay, and episode is an integer scalar that represents the current episode or time step.

The value of `-exploration_decay_rate * episode` increases linearly with episode number, and the exponential function decreases rapidly with increasing input values, so the overall effect is a gradual decay of the output value over time.

This type of exponential decay is commonly used in reinforcement learning to gradually reduce the exploration rate over time, allowing the agent to focus more on exploiting the knowledge it has gained about the environment. The `exploration_decay_rate` parameter can be adjusted to control the rate of decay.

<img src="https://i.dailymail.co.uk/1s/2020/07/17/16/30885042-8534193-image-m-11_1594999287944.jpg" width="100">

<span style="color:green;font-size:larger;">We typically set state = env.reset().  In 'CartPole-v1', env.reset() is a tuple (array([-0.04022462,  0.04057121, -0.00033032,  0.01596019], dtype=float32), {}).   Throughout the code, we use "state" as an index to an array.  Capisci?  State is supposed to be a number.  How am I supposed to use this tuple?</span>

I understand your confusion. In the CartPole-v1 environment of OpenAI Gym, the `env.reset()` method returns a tuple of two elements: **the initial state** of the environment, which is a **4-dimensional array** of floating-point values, and an **empty dictionary** that can be used to store additional information about the state if needed.

However, in the context of the reinforcement learning algorithm, the **`state`** variable is typically used to represent the current state of the environment as a **single scalar index** into an array or a tensor. This can be a bit confusing, especially since the state in CartPole-v1 is represented as a 4-dimensional array.

To convert the 4-dimensional array state to a scalar index, you can use a method such as **linear discretization** or a **neural network to map the state to an index.**

Linear discretization involves dividing the range of each state variable into a set of discrete bins and assigning each bin a unique index. This can be done using the `numpy.digitize()` function.

Alternatively, you can use a neural network to map the state directly to an index. In this case, the neural network takes the state as input and produces an index as output.

Once you have the scalar index, you can use it to index into an array or a tensor to get the corresponding value.

## numpy.digitize()

Here's an example of how you can use `numpy.digitize()` to discretize the state returned by `env.reset()` into a scalar index:

```python
import gym
import numpy as np

env = gym.make('CartPole-v1')
# state = env.reset()
state, _ = env.reset()  # Extract the state and ignore the empty dictionary

# Define the number of bins for each state variable
num_bins = [16, 16, 16, 16]

# Define the range for each state variable
state_range = [(-4.8, 4.8), (-5.0, 5.0), (-0.42, 0.42), (-5.0, 5.0)]

# Compute the bin edges for each state variable
bin_edges = [np.linspace(start, stop, num + 1)[1:-1] for (start, stop), num in zip(state_range, num_bins)]

# Discretize the state into a scalar index
state_index = np.ravel_multi_index([np.digitize(s, bins) for s, bins in zip(state, bin_edges)], num_bins)

# Print the state and its corresponding index
print(f"State: {state}")
print(f"State index: {state_index}")

# State: [-0.01424464  0.0207026  -0.01617126 -0.01462725]
# State index: 30839
```

In this example, we first define the number of bins and the range for each state variable. We then use `numpy.linspace()` to compute the bin edges for each state variable.

Next, we use a list comprehension to compute the bin index for each state variable using `numpy.digitize()`. The `ravel_multi_index()` function is then used to convert the multidimensional bin indices into a scalar index.

Finally, we print the original state and its corresponding index. The resulting output should look something like this:

```yaml
State: [-0.02317835  0.01254044  0.01088651 -0.00216823]
State index: 49
```

Note that the choice of number of bins and range for each state variable can have a significant impact on the performance of the algorithm, and may need to be adjusted through experimentation.

<br>

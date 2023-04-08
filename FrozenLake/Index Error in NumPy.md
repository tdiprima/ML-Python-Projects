In python, I'm trying to print `qtable[state]`.  But it says: "IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices"

The error message you're seeing indicates that you're trying to use an invalid index to access an element in qtable. Specifically, it seems like the state variable you're passing to the index operator is not of a valid type.

In Python, you can use square brackets to access an element of a list, tuple, or dictionary by its index. However, the error message suggests that qtable may not be a simple list, tuple, or dictionary, but rather a NumPy array.

If that's the case, then you need to make sure that the state variable you're using as an index is a valid integer, slice, or boolean array. You can use the dtype attribute of the qtable object to see what kind of data it contains.

Here's an example of how you can access an element of a NumPy array using an integer index:

```python
import numpy as np

# create a 2D array
qtable = np.array([[1, 2], [3, 4]])

# access an element by its index
state = (0, 1)
print(qtable[state[0], state[1]])  # prints 2
```

In this example, state is a tuple containing the row and column indices of the element we want to access. We use these indices to access the element of the qtable array.

If state is not a tuple but a scalar variable containing an integer, you can simply use it as an index:

```python
import numpy as np

# create a 1D array
qtable = np.array([1, 2, 3, 4])

# access an element by its index
state = 2
print(qtable[state])  # prints 3
```

In this example, `state` is a scalar variable containing the index of the element we want to access. 

We use this index to access the element of the qtable array.

ok so they're setting state to `state = environment.reset()`.  Which is `(0, {'prob': 1})`; obviously not an integer.

In the while loop, they set the state `state, reward, done, _, info = environment.step(action)`.  So...

```python
# Randomly choose an action
state = environment.action_space.sample()
```

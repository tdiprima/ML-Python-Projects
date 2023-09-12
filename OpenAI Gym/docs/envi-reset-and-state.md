## Index Error in NumPy

In python, I'm trying to print `qtable[state]`.

> IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices

```python
# visualize-agent.py
state = environment.reset()
print("environment", environment.reset())  # environment (0, {'prob': 1})

if np.max(qtable[state]) > 0: ... # eh - wrong!

# Therefore:
state = environment.reset()[0]
```

<br>

```py
print(type(state))  # <class 'int'>
print(type(qtable))  # <class 'list'>
print(type(environment.reset()))  # <class 'tuple'>
```

## Out-of-Bootstrap Error Rate

Bootstrapping Out-of-Sample Simulation

The Python code is simulating a process called bootstrapping, which is a statistical resampling technique used to estimate sampling distributions. The code is focused on computing the out-of-bootstrap percentage, which is a measure of the examples not selected during bootstrapping.

```py
from random import randint

import numpy as np

n = 1000  # number of examples

# your original set of examples
examples = set(range(n))
results = list()

for j in range(10000):
    # your bootstrapped sample
    chosen = [randint(0, n-1) for k in range(n)]

    # out-of-sample
    results.append((1000 - len(set(chosen) & examples)) / float(n))

print("\nOut-of-boostrap: %0.1f %%" % (np.mean(results) * 100))

# Out-of-boostrap: 36.8 %
```

## Here's a breakdown of what each part does:

1. Import relevant libraries:

    - Importing the `randint` function to generate random integers.

    - Importing the NumPy library for numerical operations.

2. Define variables:

    - `n = 1000`: Sets the total number of examples to 1000.

    - `examples = set(range(n))`: Creates a set of examples ranging from 0 to 999.

    - `results = list()`: Initializes an empty list to store out-of-bootstrap percentages.

3. Loop to generate bootstrapped samples:

    - The loop runs 10,000 times.

    - Within each iteration, it creates a bootstrapped sample `chosen` by randomly selecting `n` integers between 0 and `n`. Note that some integers might be picked multiple times, and some might not be picked at all.

    - It calculates the number of examples that were not selected (`1000 - len(set(chosen) & examples)`) and divides it by `n` to get a ratio. This ratio is appended to the `results` list.

4. Calculate and print the average out-of-bootstrap percentage:

    - The code computes the mean of all out-of-bootstrap percentages collected in `results` using NumPy's `np.mean()` function.

    - It then multiplies the mean by 100 to express it as a percentage and prints it.

The output will be the average percentage of examples that are not included in each bootstrap sample across 10,000 iterations.

**Note:** The line `chosen = [randint(0, n) for k in range(n)]` should probably be `chosen = [randint(0, n-1) for k in range(n)]` to be consistent with the original set of examples ranging from 0 to 999. Otherwise, the bootstrap sample could contain the integer 1000, which is not in the original set. <span style="color:lime;">&check; Updated.</span>

### ¿Por qué? set(chosen) & examples

The expression `set(chosen) & examples` is performing a set intersection between `set(chosen)` and `examples`. In Python, the `&` operator when applied to sets returns a new set containing all the elements that are common to both sets.

In this code:

- `set(chosen)`: This converts the list `chosen` into a set. This list `chosen` contains the bootstrapped sample, which means it has `n` elements selected randomly from the range `[0, n-1]` (we corrected the off-by-one error).
  
- `examples`: This is the original set of examples, containing integers from 0 to `n-1` (i.e., 0 to 999 if `n = 1000`).

So, `set(chosen) & examples` will result in a set containing all the numbers that are in both `set(chosen)` and `examples`.

After that, `len(set(chosen) & examples)` returns the number of elements in this intersection set. This gives us the count of original examples that are also present in the bootstrapped sample `chosen`.

The expression `1000 - len(set(chosen) & examples)` then calculates the number of original examples that are NOT in the bootstrapped sample. This is divided by `n` (1000) to get the ratio of out-of-bootstrap examples, which is then appended to the `results` list.

<br>

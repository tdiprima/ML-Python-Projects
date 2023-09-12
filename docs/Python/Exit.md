## Exit(1) 🧯

### Python seems to be ignoring my "exit(1)" command.

Of course put a **`print`** statement to make sure we're even getting there.

Jupyter notebooks **intercept** calls to `exit()` to prevent them from terminating the program.  (And apparently PyTorch.)

But `sys.exit(1)` would be more appropriate anyway.

All I want is the script to exit when an exception occurs.

[StackOverflow](https://stackoverflow.com/questions/438894/how-do-i-stop-a-program-when-an-exception-is-raised-in-python)

## Example with mucho info

```py
import sys

try:
    doSomethingBad()
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    print("\nType", exc_type)
    print("\nErr:", exc_obj)
    print("\nLine:", exc_tb.tb_lineno)
    sys.exit(1)
```

## Regular Example

```py
# One reason why we use "numpy" instead of "math" in Deep Learning
x = [1, 2, 3]

try:
  basic_sigmoid(x) # you will see this give an error when you run it, because x is a vector.
except Exception as ex:
  print("An exception occurred.", ex)
```

<br>

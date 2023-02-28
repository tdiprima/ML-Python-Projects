## Exit(1) 🧯

### Python seems to be ignoring my "exit(1)" command.

Of course put a print statement to make sure we're even getting there.

Jupyter notebooks intercept calls to `exit()` to prevent them from terminating the program.

But apparently, `sys.exit(1)` would be more appropriate anyway.

All I want is the script to exit when an exception occurs.

[StackOverflow](https://stackoverflow.com/questions/438894/how-do-i-stop-a-program-when-an-exception-is-raised-in-python)

```py
import sys

try:
  doSomethingBad()
except:
  print("Error: The doohickey has uncoupled from the thingamabob.")
  sys.exc_info()
  sys.exit(1)
```

<br>

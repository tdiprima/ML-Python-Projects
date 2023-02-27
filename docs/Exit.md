## Exit()

<span style="color: #000088;font-size:larger">Python seems to be ignoring my "exit(1)" command.  What is happening?</span>

The `exit(1)` command is used to terminate a Python program with a non-zero status code.

When the Python interpreter encounters this command, it immediately stops execution of the program and exits with the specified status code.

If it appears that Python is ignoring your `exit(1)` command, there could be a few possible reasons:

1. **The `exit(1)` command is not being executed:** Check that the command is actually being executed by placing a print statement just before it. If the print statement is not being executed, then the code is not being executed at all.

2. **The `exit(1)` command is being overridden:** If there is code after the `exit(1)` command, it will not be executed because the program will terminate before reaching it. However, if there is a try/except block surrounding the `exit(1)` command, the except block will still execute, which could prevent the program from terminating.

3. **The `exit(1)` command is being caught** by an external library or environment: Some libraries or environments (such as Jupyter notebooks) intercept calls to `exit()` to prevent them from terminating the program. This could cause the `exit(1)` command to be ignored.

In any case, it's a good idea to investigate why the `exit(1)` command is not working as expected.

<span style="color:red;font-size:larger">Yeah, don't do this:</span>

If you need to terminate the program forcefully, you can use `os._exit(1)` instead.

However, note that this command <mark>**immediately terminates the program without calling any cleanup functions**</mark>, so it should only be used in exceptional circumstances.


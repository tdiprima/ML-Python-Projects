## More list comprehension

<!-- How does this python code work? -->

```python
import os
from pathlib import Path

directory = "some" / "path"

classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
```

Break it down.  People do like this all the time:

```python
files = [entry.name for entry in os.scandir('.')]
# OR
files = [entry.path for entry in os.scandir('.')]
```

And then you're just adding conditions.

```python
if entry.is_dir()
# OR
if entry.is_file()
```

`entry.name` &ndash; you get that property for free.

How to access the index in 'for' loops? Easy.

```python
xs = [8, 23, 45]

for idx, x in enumerate(xs):
    print(idx, x)
```

So you read it like this:

```python
class_name: i  # This part goes together, as one thing.

# An then it's
for i, class_name in enumerate(classes)
```

<br>

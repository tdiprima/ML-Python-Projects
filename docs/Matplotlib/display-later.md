## Calculate now, display later

Yes, it is possible to generate charts using Matplotlib in a PyTorch program running on a machine with no visual display capabilities, and save them as PNG files to be displayed on a different machine.

Matplotlib has a backend system that allows you to choose how to display the generated plots, including a non-interactive backend that does not require a display. One such backend is the `Agg` backend, which is designed for creating PNG files.

You can <mark>**set the backend to `Agg`**</mark> and then save it using `plt.savefig`.

```python
import matplotlib
matplotlib.use('Agg') # set the backend to Agg

import matplotlib.pyplot as plt
import numpy as np

# generate data
x = np.arange(start=0, stop=10, step=0.1) # You can pass start, stop, and step as positional arguments as well
y = np.sin(x)

# plot the data
plt.plot(x, y)

# save the plot as a PNG file
plt.savefig('plot.png')
```

<br>
This code will generate a sine wave plot and save it as a PNG file called "plot.png" in the current working directory. You can then copy this file to a different machine and open it using an image viewer to display the plot.

<mark>**Note that the `matplotlib.use('Agg')` line should be called before importing `pyplot` or any other Matplotlib modules.**</mark>

This is because the backend needs to be set before any plotting code is executed.

<br>

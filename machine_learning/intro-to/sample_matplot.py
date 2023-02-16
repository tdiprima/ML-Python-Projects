# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# Generate a sequence of integers
x = np.arange(20)
# create a second array using sinus
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(x, y, marker="x")

# CODE NEEDS TO GO HERE TO KEEP THE PLOT OPEN
plt.show()

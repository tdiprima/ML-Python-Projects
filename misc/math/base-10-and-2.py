"""
Generates a logarithmic graph with two axes, one representing the base 10 logarithm and the other
representing the base 2 logarithm, with 100 data points between 0 and 10 on the x-axis.
"""
import matplotlib.pyplot as plt
import numpy as np

"""
So we got one x and two y's.
Linspace means we're gonna get a bunch of decimals between 0 and 10.
100, to be exact.
"""
x = np.linspace(0, 10, 100)

"""
Now, take all those values and do something with 'em!
Calculate log base 10 of all those numbers
"""
y1 = np.log10(x + 1)  # Mostly 0.x

# Calculate log base 2 of all those numbers
y2 = np.log2(x + 1)  # More whole numbers

# Plot data
plt.plot(x, y1, label='Logarithm (base 10)')
plt.plot(x, y2, label='Logarithm (base 2)')

# Add title and legend
plt.title('Logarithmic Graph (base 10 and base 2)')
plt.legend()

# Display the graph
plt.show()

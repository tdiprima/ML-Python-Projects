import matplotlib.pyplot as plt
import numpy as np


# Define the exponential function
def exp_func(x):
    return np.exp(x)


# Define the domain of the function
x_values = np.linspace(-5, 2, 100)

# Compute the values of the function and its logarithm
y_values = exp_func(x_values)
log_y_values = np.log(y_values)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the exponential function on the first subplot
ax1.plot(x_values, y_values, label='exp(x)')
ax1.legend()

# Plot the natural logarithm of the function on the second subplot
ax2.plot(y_values, log_y_values, label='log(exp(x))')
ax2.legend()

# Add titles and axis labels to the subplots
ax1.set_title('Exponential Function')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2.set_title('Natural Logarithm of Exponential Function')
ax2.set_xlabel('exp(x)')
ax2.set_ylabel('log(exp(x))')

# Show the plot
plt.show()

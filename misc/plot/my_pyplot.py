"""
Creates a simple line plot with a custom window title and plot title, then displays the plot using the matplotlib library.
"""
import matplotlib.pyplot as plt

# create a plot
plt.plot([1, 2, 3], [4, 5, 6])

# set the title of the figure window
figure = plt.gcf()
# figure.canvas.set_window_title('My Custom Title')
figure.canvas.manager.set_window_title('My Window Title')

# set the title of the plot
plt.title('My Plot Title')

# display the plot
plt.show()

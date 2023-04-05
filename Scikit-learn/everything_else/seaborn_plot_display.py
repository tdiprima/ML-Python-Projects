"""
Unfortunately, there is no show() method for Seaborn plots.
Then what the hecc is this?  seaborn.objects.Plot.show
In Seaborn, you can use the matplotlib.pyplot functions to show your plot.
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Create a Seaborn plot
sns.set(style="darkgrid")
tips = sns.load_dataset("tips")
ax = sns.countplot(x="day", data=tips)

# Show the plot using pyplot
plt.show()

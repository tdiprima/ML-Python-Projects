"""
Loads the "tips" dataset from seaborn library
Creates a count plot of the 'day' column using seaborn
Displays it using matplotlib
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Create a Seaborn plot
sns.set(style="darkgrid")
tips = sns.load_dataset("tips")
ax = sns.countplot(x="day", data=tips)

# Show the plot using pyplot
plt.show()

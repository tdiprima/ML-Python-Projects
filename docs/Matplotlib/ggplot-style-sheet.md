## Python Plotting with Matplotlib

<img width="600" src="https://matplotlib.org/stable/_images/sphx_glr_ggplot_001_2_00x.png" />

[matplotlib.org](https://matplotlib.org/stable/gallery/style_sheets/ggplot.html)

```python
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

# Fixing random state for reproducibility
np.random.seed(19680801)

fig, axs = plt.subplots(ncols=2, nrows=2)
ax1, ax2, ax3, ax4 = axs.flat

# scatter plot (Note: `plt.scatter` doesn't use default colors)
x, y = np.random.normal(size=(2, 200))
ax1.plot(x, y, 'o')

# sinusoidal lines with colors from default color cycle
L = 2 * np.pi
x = np.linspace(0, L)

ncolors = len(plt.rcParams['axes.prop_cycle'])
shift = np.linspace(0, L, ncolors, endpoint=False)

for s in shift:
    ax2.plot(x, np.sin(x + s), '-')

ax2.margins(0)

# bar graphs
x = np.arange(5)
y1, y2 = np.random.randint(1, 25, size=(2, 5))
width = 0.25

ax3.bar(x, y1, width)
ax3.bar(x + width, y2, width,
        color=list(plt.rcParams['axes.prop_cycle'])[2]['color'])
ax3.set_xticks(x + width, labels=['a', 'b', 'c', 'd', 'e'])

# circles with colors from default color cycle
for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
    xy = np.random.normal(size=2)
    ax4.add_patch(plt.Circle(xy, radius=0.3, color=color['color']))

ax4.axis('equal')
ax4.margins(0)

plt.show()
```

<br>
Sure, this script is generating a 2x2 grid of plots using the matplotlib and numpy libraries in Python. Here's what each part of the code is doing:

- The script starts by importing the necessary **libraries**, matplotlib and numpy. The `matplotlib.pyplot` library is used for creating static, animated, and interactive visualizations in Python. The `numpy` library is used for working with arrays.

- The `plt.style.use('ggplot')` sets the **style** of the plots to resemble those in ggplot2 (a popular plotting package for R).

- `np.random.seed(19680801)` sets the random number generator **seed** to a specific number. This ensures the randomness in the script is repeatable. The number itself doesn't matter, but if you run the script multiple times with the same seed, you'll get the same output each time.

- `fig, axs = plt.subplots(ncols=2, nrows=2)` generates a 2x2 **grid** of subplots and assigns their **Axes** to `axs`.

    `ax1, ax2, ax3, ax4 = axs.flat` is then used to assign these individual **Axes objects** to `ax1, ax2, ax3, ax4` respectively.

- The script then generates two 200-element arrays of random numbers from a **normal** distribution using `np.random.normal(size=(2, 200))`.

    It plots these against each other on the first subplot <span style="color:#0f0;font-weight:bold;">(ax1)</span> as points ('o').

- On the second subplot <span style="color:#0ff;font-weight:bold;">(ax2)</span>, the script plots a series of **sine waves**, each of which is phase-shifted according to the value `s` in the array `shift`.

- On the third subplot <span style="color:#00f;font-weight:bold;">(ax3)</span>, it generates two sets of five random integers between 1 and 25 and plots them as a pair of side-by-side **bar graphs.**

- The fourth subplot <span style="color:#997fff;font-weight:bold;">(ax4)</span> adds a number of **colored circles** at random locations, with colors defined by the plot style's color cycle.

- `plt.show()` finally displays the figure with all the subplots.

This is a complex script that showcases some of the wide range of plotting capabilities of matplotlib, including scatter plots, line plots, bar plots, and manual placement of shapes.

<br>


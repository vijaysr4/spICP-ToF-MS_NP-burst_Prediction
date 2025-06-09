import pandas as pd
import matplotlib.pyplot as plt
import os

# Load & sort
df = pd.read_csv("NPs_BHVO_Oct23_full.csv").sort_values("Time (ms)")

# Identify element columns
elements = df.select_dtypes(include="number").columns.drop("Time (ms)")

# Prepare subplot grid
n = len(elements)
cols = 4
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(25, rows * 3), sharex=True)
axes = axes.flatten()

# Choose a distinct color for each element
colors = plt.cm.tab20.colors  # up to 20 distinct colors
color_cycle = [colors[i % len(colors)] for i in range(n)]

# Plot each element in its own subplot
for ax, el, col in zip(axes, elements, color_cycle):
    ax.plot(df["Time (ms)"], df[el], color=col, linewidth=1)
    ax.set_title(el, fontsize=11)
    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.5)

# Turn off unused axes
for ax in axes[len(elements):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("elements_subplots_color.png", dpi=200)
plt.show()

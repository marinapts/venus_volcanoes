from src.pyvov import ChipsIndex
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from data_loader import DataLoader

data_loader = DataLoader()
labels = data_loader.get_labels()

categories = [0, 1, 2, 3, 4]
frequencies = np.bincount(labels)

# Set grid style
sns.set(style="whitegrid")

fig, ax = plt.subplots()

# Create the bar plot
ax = sns.barplot(x=categories, y=frequencies, color="red", saturation=.5)
# ax = sns.countplot(x=frequencies, color="red", saturation=.5, orient='v')
total = len(labels)

# Display the values above the bars
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:d}'.format(int(height)),
            ha="center")

ax.set_title('Class Label Count')
ax.set_xlabel('Category')

# Display the plot
plt.show()

# Uncomment the line below to save the figure as vector image
# fig.savefig('figures/class-imbalance-histogram.pdf')
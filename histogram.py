from src.pyvov import ChipsIndex
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from data_loader import DataLoader

data_loader = DataLoader()
labels = data_loader.get_labels()

categories = [1, 2, 3, 4, 5]
frequencies = np.bincount(labels)

# Set grid style
sns.set(style="whitegrid")

fig, ax = plt.subplots()

# Create the bar plot
ax = sns.barplot(x=categories, y=frequencies, color="red", saturation=.5)
ax.set_title('Class Label Count')
ax.set_xlabel('Category')

# Display the plot
plt.show()

# Uncomment the line below to save the figure as vector image
# fig.savefig('nos-plot-fixed.pdf')
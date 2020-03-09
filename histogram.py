from src.pyvov import ChipsIndex
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

ci = ChipsIndex()

all_experiments = ci.experiments()
EXP_NAMES = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'C1', 'D1', 'D2', 'D3', 'D4', 'E1', 'E2', 'E3', 'E4', 'E5']
num_img = 0
training_split = []
testing_split = []
all_labels = []

for EXP_NAME in ['C1', 'D4']:
    training_split.extend(ci.training_split_for(EXP_NAME))
    testing_split.extend(ci.testing_split_for(EXP_NAME))
    labels = ci.labels_for(EXP_NAME)
    label_list = list(labels['trn'])
    label_list.extend(list(labels['tst']))
    all_labels.extend(label_list)

training_split.extend(testing_split)
full_dataset = training_split

categories = [1, 2, 3, 4, 5]
frequencies = np.bincount(all_labels)

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
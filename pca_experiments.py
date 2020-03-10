from data_loader import DataLoader
import sklearn.decomposition
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from math import isnan


def normalise(array):
    normalised_array = array - np.mean(array)
    normalised_array = normalised_array / np.std(normalised_array)
    return normalised_array


seed = 8

# Load the data
data_loader = DataLoader()
full_dataset = data_loader.get_full_dataset()
labels = data_loader.get_labels()

# Normalise each image
means = [np.mean(image) for image in full_dataset]
stds = [np.std(image) for image in full_dataset]

# Indices of constant images
indices = [i for i in np.arange(len(full_dataset)) if np.std(full_dataset[i]) == 0]

# Remove constant images
full_dataset = [full_dataset[i] for i in np.arange(len(full_dataset)) if i not in indices]
labels = [labels[i] for i in np.arange(len(labels)) if i not in indices]

# Normalise the remaining data
full_dataset = [normalise(image) for image in full_dataset]

# Only consider positive examples
non_zero_data = [full_dataset[i] for i in np.arange(len(labels)) if labels[i] != 0]
non_zero_labels = [label for label in labels if label != 0]

# Perform 2-dimensional projection via PCA
pca = sklearn.decomposition.PCA(n_components=2, random_state=seed)
transformed_data = pca.fit_transform(non_zero_data)

# Create a scatter plot using the reduced features
# ax = sns.scatterplot(x=transformed_data[:, 0], y=transformed_data[:, 1], hue=labels)
labels = non_zero_labels
fig, ax = plt.subplots(1)
for label in np.unique(labels):
    x = [transformed_data[i, 0] for i in np.arange(len(transformed_data)) if labels[i] == label]
    y = [transformed_data[i, 1] for i in np.arange(len(transformed_data)) if labels[i] == label]
    if label == 0:
        ax.scatter(x, y, label=label, color='navajowhite')
    else:
        ax.scatter(x, y, label=label)


ax.legend()
ax.set_title('Two-dimensional data projection via PCA on positive examples')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
# plt.legend()
plt.show()

# Uncomment the line below to save the figure as vector image
# fig.savefig('pca-positive-examples.pdf')

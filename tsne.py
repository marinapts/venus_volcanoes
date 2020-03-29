from data_loader import DataLoader
import sklearn.decomposition
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from pca_experiments import normalise

seed = 8
data_loader = DataLoader()
full_dataset = data_loader.get_full_dataset()
labels = data_loader.get_labels()

# Indices of constant images
indices = [i for i in np.arange(len(full_dataset)) if np.std(full_dataset[i]) == 0]

# Remove constant images
full_dataset = [full_dataset[i] for i in np.arange(len(full_dataset)) if i not in indices]
labels = [labels[i] for i in np.arange(len(labels)) if i not in indices]

# Normalise the remaining data
full_dataset = [normalise(image) for image in full_dataset]

non_zero_data = [full_dataset[i] for i in np.arange(len(labels)) if labels[i] != 0]
non_zero_labels = [label for label in labels if label != 0]

# pca = sklearn.decomposition.PCA(n_components=2, random_state=seed)
# transformed_data = pca.fit_transform(non_zero_data)

# ax = sns.scatterplot(x=transformed_data[:, 0], y=transformed_data[:, 1], hue=non_zero_labels)
# plt.show()


# Perform t-SNE and plot the results for different perplexities
fig, ax = plt.subplots(3, 2, figsize=(12, 14))
for ii, perplexity in enumerate([2, 5, 10, 30, 50, 100]):
    print('Performing TSNE with perplexity {}'.format(perplexity))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=10)
    transformed_data = tsne.fit_transform(non_zero_data)
    plt.subplot(3, 2, ii + 1)
    ax = sns.scatterplot(x=transformed_data[:, 0], y=transformed_data[:, 1], hue=non_zero_labels)
    plt.title('Perplexity: {}, KL-score: {}'.format(perplexity, tsne.kl_divergence_))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2 ')
plt.legend(loc='center left', bbox_to_anchor=[1.01, 1.5], scatterpoints=3)
# fig.suptitle('t-SNE Projections on the positive examples')
fig.tight_layout()
plt.show()

# Uncomment the line below to save the figure as vector image
# fig.savefig('figures/tsne-positive-examples.pdf')

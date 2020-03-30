from data_loader import DataLoader
import sklearn.decomposition
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from pca_experiments import normalise
from pca_experiments import load_data, normalize_ds

def tsne(full_dataset, labels):
    # Perform t-SNE and plot the results for different perplexities
    fig, ax = plt.subplots(3, 2, figsize=(12, 14))
    for ii, perplexity in enumerate([2, 5, 10, 30, 50, 100]):
        print('Performing TSNE with perplexity {}'.format(perplexity))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=10)
        transformed_data = tsne.fit_transform(full_dataset)
        plt.subplot(3, 2, ii + 1)
        ax = sns.scatterplot(x=transformed_data[:, 0], y=transformed_data[:, 1], hue=labels)
        plt.title('Perplexity: {}, KL-score: {}'.format(perplexity, tsne.kl_divergence_))
        plt.xlabel('Component 1')
        plt.ylabel('Component 2 ')
    plt.legend(loc='center left', bbox_to_anchor=[1.01, 1.5], scatterpoints=3)
    # fig.suptitle('t-SNE Projections on the positive examples')
    fig.tight_layout()
    fig.savefig('figures/tsne.pdf')


if __name__ == '__main__':
    seed = 8
    full_dataset, labels = load_data()
    full_dataset, labels = normalize_ds(full_dataset, labels)
    smaller_dataset = full_dataset[0:5000]
    smaller_labels = labels[0:5000]
    tsne(smaller_dataset, smaller_labels)

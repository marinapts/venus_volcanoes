from reduce_dimensionality import reduce_dims, load_data, normalize_ds
from data_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


colordict = {0: 'navajowhite', 1: 'midnightblue', 2: 'royalblue' , 3: 'cornflowerblue', 4: 'lightskyblue'}

def umap_plot(transformed_dataset, labels):
    # Plot first 2 components
    fig, ax = plt.subplots(1)
    for label in np.unique(labels):
        ix = np.where(labels == label)
        plt.scatter(transformed_dataset[ix, 0], transformed_dataset[ix, 1], color = colordict[label], label=label)
    ax.legend()
    ax.set_title('Two-dimensional data projection via UMAP')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    fig.savefig('figures/umap.pdf')

def pca_plots(transformed_data, labels):
    # Decide colors for each label
    colordict = {0: 'navajowhite', 1: 'midnightblue', 2: 'royalblue' , 3: 'cornflowerblue', 4: 'lightskyblue'}

    # Plot first 2 components
    fig, ax = plt.subplots(1)
    for label in np.unique(labels):
        ix = np.where(labels == label)
        plt.scatter(transformed_data[ix, 0], transformed_data[ix, 1], color = colordict[label], label=label)
    ax.legend()
    ax.set_title('Two-dimensional data projection via PCA')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    fig.savefig('figures/pca_2dim.pdf')

    # Plot first 3 components
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    for label in np.unique(labels):
        ix = np.where(labels == label)
        ax.scatter(transformed_data[ix, 0], transformed_data[ix, 1], transformed_data[ix, 2], color = colordict[label], label=label)
    ax.legend()
    ax.set_title('Two-dimensional data projection via PCA')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.figure.savefig('figures/pca_3dim.pdf')

def tsne(full_dataset, labels, seed):
    # Perform t-SNE and plot the results for different perplexities
    fig, ax = plt.subplots(3, 2, figsize=(12, 14))
    for ii, perplexity in enumerate([2, 5, 10, 30, 50, 100]):
        print('Performing TSNE with perplexity {}'.format(perplexity))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
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

def lle_plot(transformed_dataset, labels):
    # Plot first 2 components
    fig, ax = plt.subplots(1)
    for label in np.unique(labels):
        ix = np.where(labels == label)
        plt.scatter(transformed_dataset[ix, 0], transformed_dataset[ix, 1], color = colordict[label], label=label)
    ax.legend()
    ax.set_title('Two-dimensional data projection via LLE')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    fig.savefig('figures/lle.pdf')

def mds_plot(transformed_dataset, labels):
    # Plot first 2 components
    fig, ax = plt.subplots(1)
    for label in np.unique(labels):
        ix = np.where(labels == label)
        plt.scatter(transformed_dataset[ix, 0], transformed_dataset[ix, 1], color = colordict[label], label=label)
    ax.legend()
    ax.set_title('Two-dimensional data projection via metric MDS')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    fig.savefig('figures/metricMDS.pdf')

def isomap_plot(transformed_dataset, labels):
    # Plot first 2 components
    fig, ax = plt.subplots(1)
    for label in np.unique(labels):
        ix = np.where(labels == label)
        plt.scatter(transformed_dataset[ix, 0], transformed_dataset[ix, 1], color = colordict[label], label=label)
    ax.legend()
    ax.set_title('Two-dimensional data projection via ISOMAP')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    fig.savefig('figures/isomap.pdf')



if __name__ == '__main__':

    seed = 8
    num_comps = 2
    full_dataset, labels = load_data()
    full_dataset, labels = normalize_ds(full_dataset, labels)

    smaller_dataset = np.array(full_dataset)[0:5000,:]
    smaller_labels = labels[0:5000]
    #full_dataset, labels = normalize_ds(full_dataset, labels)

    #PCA
    #reduced_data = reduce_dims('pca', full_dataset, 3, seed)
    #pca_plots(reduced_data, labels)
    #print(reduced_data.shape)

    #UMAP
    #reduced_data = reduce_dims('umap', smaller_dataset, num_comps, seed)
    #umap_plot(reduced_data, smaller_labels)
    #print(reduced_data.shape)

    #T-SNE
    #tsne(smaller_dataset, smaller_labels, seed)

    #LLE:
    reduced_data = reduce_dims('lle', smaller_dataset, num_comps, seed, num_neighbors=5)
    lle_plot(reduced_data, smaller_labels)
    print(reduced_data.shape)

    smaller_dataset = np.array(full_dataset)[0:1000,:]
    smaller_labels = labels[0:1000]

    #MDS:
    reduced_data = reduce_dims('mds', smaller_dataset, num_comps, seed)
    mds_plot(reduced_data, smaller_labels)
    print(reduced_data.shape)

    #Isomap:
    reduced_data = reduce_dims('isomap', smaller_dataset, num_comps, seed)
    isomap_plot(reduced_data, smaller_labels)
    print(reduced_data.shape)

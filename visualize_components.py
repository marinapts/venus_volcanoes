from reduce_dimensionality import reduce_dims, load_data, normalize_ds
from data_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


colordict = {0: 'navajowhite', 1: 'royalblue', 2: 'cornflowerblue' , 3: 'lightsteelblue', 4: 'slategrey'}

label_dict = {0: 'no volcano', 1: 'certainly', 2: 'probably', 3: 'possibly', 4: 'pit'}

def umap_plot(transformed_dataset, labels):
    # Plot first 2 components
    fig, ax = plt.subplots(1)
    for label in np.unique(labels):
        ix = np.where(labels == label)
        plt.scatter(transformed_dataset[ix, 0], transformed_dataset[ix, 1], color = colordict[label], label=label_dict[label])
    ax.legend()
    #ax.set_title('Two-dimensional data projection via PCA')
    ax.set_xlabel('First Dimension',fontsize=15)
    ax.set_ylabel('Second Dimension',fontsize=15)
    ax.tick_params(labelsize=15)
    fig.savefig('figures/umap.pdf', format = 'pdf', bbox_inches = 'tight')

def pca_plots(transformed_data, labels):
    # Plot first 2 components
    fig, ax = plt.subplots(1)
    for label in np.unique(labels):
        ix = np.where(labels == label)
        plt.scatter(transformed_data[ix, 0], transformed_data[ix, 1], color = colordict[label], label=label_dict[label])
    ax.legend()
    #ax.set_title('Two-dimensional data projection via PCA')
    ax.set_xlabel('First Dimension',fontsize=15)
    ax.set_ylabel('Second Dimension',fontsize=15)
    ax.tick_params(labelsize=15)
    fig.tight_layout()
    fig.savefig('figures/pca_2dim.pdf', format = 'pdf', bbox_inches = 'tight')

    # Plot first 3 components
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    for label in np.unique(labels):
        ix = np.where(labels == label)
        ax.scatter(transformed_data[ix, 0], transformed_data[ix, 1], transformed_data[ix, 2], color = colordict[label], label=label_dict[label])
    ax.legend()
    #ax.set_title('Two-dimensional data projection via PCA')
    ax.set_xlabel('First Dimension',fontsize=15)
    ax.set_ylabel('Second Dimension',fontsize=15)
    ax.tick_params(labelsize=15)
    ax.set_zlabel('Component 3')
    ax.figure.savefig('figures/pca_3dim.pdf', format = 'pdf', bbox_inches = 'tight')

def tsne(full_dataset, labels, seed):
    colordict = {0: 'navajowhite', 1: 'navy', 2: 'blue' , 3: 'turquoise', 4: 'coral'}
    # Perform t-SNE and plot the results for different perplexities
    fig, ax = plt.subplots(3, 2, figsize=(12, 14))
    for ii, perplexity in enumerate([2, 5, 10, 30, 50, 100]):
        print('Performing TSNE with perplexity {}'.format(perplexity))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
        transformed_data = tsne.fit_transform(full_dataset)
        plt.subplot(3, 2, ii + 1)
        for label in np.unique(labels):
            ix = np.where(labels == label)
            #plt.scatter(transformed_dataset[ix, 0], transformed_dataset[ix, 1], color = colordict[label], label=label_dict[label])
            plt.scatter(x=transformed_data[ix, 0], y=transformed_data[ix, 1], color=colordict[label], label=label_dict[label])
        plt.title('Perplexity: {}, KL-score: {}'.format(perplexity, tsne.kl_divergence_))
        plt.xlabel('First Dimension',fontsize=15)
        plt.ylabel('Second Dimension ',fontsize=15)
        plt.tick_params(labelsize=15)
        plt.legend()
    # fig.suptitle('t-SNE Projections on the positive examples')
    fig.tight_layout()
    fig.savefig('figures/tsne.pdf', format = 'pdf', bbox_inches = 'tight')

def lle_plot(transformed_dataset, labels):
    # Plot first 2 components
    fig, ax = plt.subplots(1)
    for label in np.unique(labels):
        ix = np.where(labels == label)
        plt.scatter(transformed_dataset[ix, 0], transformed_dataset[ix, 1], color = colordict[label], label=label_dict[label])
    ax.legend()
    #ax.set_title('Two-dimensional data projection via LLE')
    ax.set_xlabel('First Dimension',fontsize=15)
    ax.set_ylabel('Second Dimension',fontsize=15)
    ax.tick_params(labelsize=15)
    fig.tight_layout()
    fig.savefig('figures/lle.pdf', format = 'pdf', bbox_inches = 'tight')

def mds_plot(transformed_dataset, labels):
    # Plot first 2 components
    fig, ax = plt.subplots(1)
    for label in np.unique(labels):
        ix = np.where(labels == label)
        plt.scatter(transformed_dataset[ix, 0], transformed_dataset[ix, 1], color = colordict[label], label=label_dict[label])
    ax.legend()
    #ax.set_title('Two-dimensional data projection via metric MDS')
    ax.set_xlabel('First Dimension',fontsize=15)
    ax.set_ylabel('Second Dimension',fontsize=15)
    ax.tick_params(labelsize=15)
    fig.tight_layout()
    fig.savefig('figures/metricMDS.pdf', format = 'pdf', bbox_inches = 'tight')

def isomap_plot(transformed_dataset, labels):
    # Plot first 2 components
    fig, ax = plt.subplots(1)
    for label in np.unique(labels):
        ix = np.where(labels == label)
        plt.scatter(transformed_dataset[ix, 0], transformed_dataset[ix, 1], color = colordict[label], label=label_dict[label])
    ax.legend()
    #ax.set_title('Two-dimensional data projection via ISOMAP')
    ax.set_xlabel('First Dimension',fontsize=15)
    ax.set_ylabel('Second Dimension',fontsize=15)
    ax.tick_params(labelsize=15)
    fig.tight_layout()
    fig.savefig('figures/isomap.pdf', format = 'pdf', bbox_inches = 'tight')

def all_plot(full_dataset, labels, seed):
    colordict = {0: 'navajowhite', 1: 'navy', 2: 'blue' , 3: 'turquoise', 4: 'coral'}
    # Perform t-SNE and plot the results for different perplexities
    fig, ax = plt.subplots(3, 2, figsize=(12, 14))
    for ii, perplexity in methods_list:
        print('Performing TSNE with perplexity {}'.format(perplexity))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
        transformed_data = tsne.fit_transform(full_dataset)
        plt.subplot(3, 2, ii + 1)
        for label in np.unique(labels):
            ix = np.where(labels == label)
            #plt.scatter(transformed_dataset[ix, 0], transformed_dataset[ix, 1], color = colordict[label], label=label_dict[label])
            plt.scatter(x=transformed_data[ix, 0], y=transformed_data[ix, 1], color=colordict[label], label=label_dict[label])
        plt.title('Perplexity: {}, KL-score: {}'.format(perplexity, tsne.kl_divergence_))
        plt.xlabel('First Dimension',fontsize=15)
        plt.ylabel('Second Dimension ',fontsize=15)
        plt.tick_params(labelsize=15)
        plt.legend()
    # fig.suptitle('t-SNE Projections on the positive examples')
    fig.tight_layout()
    fig.savefig('figures/tsne.pdf', format = 'pdf', bbox_inches = 'tight')




if __name__ == '__main__':

    seed = 8
    num_comps = 2
    full_dataset, labels = load_data()
    full_dataset, labels = normalize_ds(full_dataset, labels)

    smaller_dataset = np.array(full_dataset)[0:5000,:]
    smaller_labels = labels[0:5000]
    #full_dataset, labels = normalize_ds(full_dataset, labels)

    # # #PCA
    # reduced_data = reduce_dims('pca', full_dataset, labels, 3, seed, only_positives=False)
    # pca_plots(reduced_data, labels)
    # print(reduced_data.shape)

    # # #UMAP
    # reduced_data = reduce_dims('umap', smaller_dataset, smaller_labels, num_comps, seed, only_positives=False)
    # umap_plot(reduced_data, smaller_labels)
    # print(reduced_data.shape)

    # #T-SNE
    # positives = np.array(full_dataset)[np.where(np.array(labels)>0)[0],:]
    # positivelabels = np.array(labels)[np.where(np.array(labels)>0)[0]]
    # tsne(positives, positivelabels, seed)

    #LLE:
    reduced_data = reduce_dims('lle', smaller_dataset, smaller_labels, num_comps, seed, num_neighbors=5, only_positives=False)
    lle_plot(reduced_data, smaller_labels)
    print(reduced_data.shape)

    #smaller_dataset = np.array(smaller_dataset)[0:1000,:]
    #smaller_labels = smaller_labels[0:1000]

    #MDS:
    reduced_data = reduce_dims('mds', smaller_dataset, smaller_labels, num_comps, seed, only_positives=False)
    print(reduced_data.shape, len(smaller_labels))
    mds_plot(reduced_data, smaller_labels)
    print(reduced_data.shape)

    #Isomap:
    reduced_data = reduce_dims('isomap', smaller_dataset, smaller_labels, num_comps, seed, only_positives=False)
    isomap_plot(reduced_data, smaller_labels)
    print(reduced_data.shape)

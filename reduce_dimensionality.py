from sklearn.manifold import TSNE
import umap
import numpy as np
from data_loader import DataLoader
import sklearn.decomposition
from sklearn.manifold import LocallyLinearEmbedding, MDS, Isomap

def load_data():
    # Load the data
    data_loader = DataLoader()
    full_dataset, labels = data_loader.get_full_dataset()
    return full_dataset, labels

def normalise(array):
    normalised_array = array - np.mean(array)
    normalised_array = normalised_array / np.std(normalised_array)
    return normalised_array

def normalize_ds(full_dataset, labels):
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

    return full_dataset, labels

def pca_transform(dataset, num_comps, seed, perplexity=None):
    # Perform 2-dimensional projection via PCA
    pca = sklearn.decomposition.PCA(n_components=num_comps, random_state=seed)
    fit_positives = pca.fit(dataset)
    return fit_positives

def tsne_transform(dataset, num_comps, perplexity, seed):
    tsne = TSNE(n_components=num_comps, random_state=seed, perplexity=perplexity)
    fit_positives = tsne.fit(dataset)
    return fit_positives

def umap_transform(dataset, num_comps, seed):
    embedding = umap.UMAP(n_components=num_comps, random_state=seed)
    fit_positives = embedding.fit(dataset)
    return fit_positives

def locally_linear_embedding(dataset, num_comps, seed, num_neighbors=5):
    embedding = LocallyLinearEmbedding(n_components=num_comps, random_state=seed, n_neighbors=num_neighbors)
    fit_positives = embedding.fit(dataset)
    return fit_positives

def multi_dim_scaling(dataset, num_comps, seed, num_neighbors=5, metric=True):
    embedding = MDS(n_components=num_comps, random_state=seed)
    fit_positives = embedding.fit(dataset)
    return fit_positives

def isomap_transform(dataset, num_comps, seed, num_neighbors=5):
    embedding = Isomap(n_components=num_comps, n_neighbors=num_neighbors)
    fit_positives = embedding.fit(dataset)
    return fit_positives

def reduce_dims(type_transform, dataset, labels, num_comps, seed, perplexity=None, num_neighbors=None, metric=True):
    positives = np.array(dataset)[np.where(np.array(labels)>0)[0],:]
    if type_transform == 'pca':
        fit_positives = pca_transform(positives, num_comps, seed, perplexity=None)
    elif type_transform == 'umap':
        fit_positives = umap_transform(positives, num_comps, seed)
    elif type_transform == 'lle':
        fit_positives = locally_linear_embedding(positives, num_comps, seed, num_neighbors)
    elif type_transform == 'mds':
        fit_positives = multi_dim_scaling(positives, num_comps, seed)
    elif type_transform == 'isomap':
        fit_positives = isomap_transform(positives, num_comps, seed)
    transformed_data = fit_positives.transform(dataset)
    return transformed_data

if __name__ == '__main__':

    seed = 8
    num_comps = 6
    full_dataset, labels = load_data()
    full_dataset, labels = normalize_ds(full_dataset, labels)

    #PCA
    reduced_data = reduce_dims('pca', full_dataset, labels, num_comps, seed)
    print(reduced_data.shape)

    #UMAP
    reduced_data = reduce_dims('umap', full_dataset, labels, num_comps, seed)
    print(reduced_data.shape)

    #LLE:
    reduced_data = reduce_dims('lle', full_dataset, labels, num_comps, seed, num_neighbors=5)
    print(reduced_data.shape)

    #MDS:
    reduced_data = reduce_dims('mds', full_dataset, labels, num_comps, seed, metric=True)
    print(reduced_data.shape)

    #Isomap:
    reduced_data = reduce_dims('isomap', full_dataset, labels, num_comps, seed)
    print(reduced_data.shape)







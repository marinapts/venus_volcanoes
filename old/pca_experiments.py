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

def load_data():
    # Load the data
    data_loader = DataLoader()
    full_dataset = data_loader.get_full_dataset()
    labels = data_loader.get_labels()
    return full_dataset, labels

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

def pca_transform(full_dataset, labels, num_comps, seed):
    # Perform 2-dimensional projection via PCA
    pca = sklearn.decomposition.PCA(n_components=num_comps, random_state=seed)
    transformed_data = pca.fit_transform(full_dataset)
    return transformed_data

def pca_plots(full_dataset, labels, seed):
    #Perform PCA
    transformed_data = pca_transform(full_dataset, labels, 6, seed)

    # Decide colors for each label
    colordict = {0: 'navajowhite', 1: 'midnightblue', 2: 'royalblue' , 3: 'cornflowerblue', 4: 'lightskyblue'}

    # Plot first 2 components
    fig, ax = plt.subplots(1)
    for label in np.unique(labels):
        ix = np.where(labels == label)
        plt.scatter(transformed_data[ix, 0], transformed_data[ix, 1], color = colordict[label], label=label)
    ax.legend()
    ax.set_title('Two-dimensional data projection via PCA on positive examples')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    fig.savefig('figures/pca_2dim.pdf')

    # Plot first 3 components
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    for label in np.unique(labels):
        ix = np.where(labels == label)
        ax.scatter(transformed_data[ix, 0], transformed_data[ix, 1], transformed_data[ix, 2], color = colordict[label], label=label)
    ax.legend()
    ax.set_title('Two-dimensional data projection via PCA on positive examples')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    ax.figure.savefig('figures/pca_3dim.pdf')

if __name__ == '__main__':
    seed = 8
    full_dataset, labels = load_data()
    full_dataset, labels = normalize_ds(full_dataset, labels)

    # Only consider positive examples
    #non_zero_data = [full_dataset[i] for i in np.arange(len(labels)) if labels[i] != 0]
    #non_zero_labels = [label for label in labels if label != 0]

    pca_plots(full_dataset, labels, seed)


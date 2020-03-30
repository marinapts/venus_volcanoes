import umap
from pca_experiments import load_data, normalize_ds
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    seed = 8
    full_dataset, labels = load_data()
    full_dataset, labels = normalize_ds(full_dataset, labels)
    smaller_dataset = full_dataset[0:5000]
    smaller_labels = labels[0:5000]
    full_dataset = smaller_dataset
    labels = smaller_labels

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(full_dataset)
    print(embedding.shape)

    colordict = {0: 'navajowhite', 1: 'midnightblue', 2: 'royalblue' , 3: 'cornflowerblue', 4: 'lightskyblue'}

    # Plot first 2 components
    fig, ax = plt.subplots(1)
    for label in np.unique(labels):
        ix = np.where(labels == label)
        plt.scatter(embedding[ix, 0], embedding[ix, 1], color = colordict[label], label=label)
    ax.legend()
    ax.set_title('Two-dimensional data projection via UMAP')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    fig.savefig('figures/umap.pdf')


    #plt.scatter(embedding[:, 0], embedding[:, 1]) #, c=[sns.color_palette()[x] for x in iris.target])
    #plt.gca().set_aspect('equal', 'datalim')
    #plt.title('UMAP projection of the Iris dataset', fontsize=24)
    plt.show()

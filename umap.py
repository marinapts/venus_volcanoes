import umap
from pca_experiments import load_data, normalize_ds
import matplotlib.pyplot as plt

if __name__ == '__main__':

    seed = 8
    full_dataset, labels = load_data()
    full_dataset, labels = normalize_ds(full_dataset, labels)

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(full_dataset[0:1000])
    print(embedding.shape)

    plt.scatter(embedding[:, 0], embedding[:, 1]) #, c=[sns.color_palette()[x] for x in iris.target])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Iris dataset', fontsize=24)
    plt.show()

from data_loader import DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter


def plot_cluster_centers(centers):
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(centers[0].reshape(15, 15), cmap='Greys')
    axs[0, 0].set_title('Almost certainly a volcano')
    axs[0, 1].imshow(centers[1].reshape(15, 15), cmap='Greys')
    axs[0, 1].set_title('Probably a volcano')
    axs[1, 0].imshow(centers[2].reshape(15, 15), cmap='Greys')
    axs[1, 0].set_title('Possibly a volcano')
    axs[1, 1].imshow(centers[3].reshape(15, 15), cmap='Greys')
    axs[1, 1].set_title('A pit')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.show()


if __name__ == '__main__':
    data = DataLoader()
    X_train_volcanoes, y_train_volcanoes = data.get_volcanoes_training_set()
    print('Class distribution before clustering:', Counter(y_train_volcanoes))

    kmeans = KMeans(n_clusters=4, random_state=0).fit(X_train_volcanoes)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    print('Class distribution after clustering:', Counter(cluster_labels))
    print('centers:', len(cluster_centers))

    plot_cluster_centers(cluster_centers)

    # Apply PCA to each cluster
    # pca = PCA(n_components=2)
    # pca.fit(X_train_volcanoes)

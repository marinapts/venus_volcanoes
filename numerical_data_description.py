from pca_experiments import load_data, normalize_ds
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def remove_black_images(full_dataset, labels):
    # Normalise each image
    means = [np.mean(image) for image in full_dataset]
    stds = [np.std(image) for image in full_dataset]

    # Indices of constant images
    indices = [i for i in np.arange(len(full_dataset)) if np.std(full_dataset[i]) == 0]

    # Remove constant images
    full_dataset = [full_dataset[i] for i in np.arange(len(full_dataset)) if i not in indices]
    labels = [labels[i] for i in np.arange(len(labels)) if i not in indices]

    # Normalise the remaining data
    full_dataset = [image for image in full_dataset]

    return full_dataset, labels

def hist_by_class(full_dataset, labels):
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    mean_per_image = np.mean(full_dataset, axis = 1)
    plt.subplot(3, 2, 1)
    ax = plt.hist(mean_per_image)
    plt.title('Full dataset')

    for ii, label in enumerate(np.unique(labels)):
        ix = np.where(labels == label)
        mean_per_image = np.mean(full_dataset[list(ix[0]), :], axis = 1)
        plt.subplot(3, 2, ii + 2)
        ax = plt.hist(mean_per_image)
        plt.title('Class: {}'.format(label))
    fig.savefig('figures/hist_classes.pdf')


def mean_image_by_class(full_dataset, labels):
    #Mean image by class
    full_dataset = np.array(full_dataset)
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    for ii, label in enumerate(np.unique(labels)):
        ix = np.where(labels == label)
        mean_image = np.mean(full_dataset[list(ix[0]), :], axis = 0)
        plt.subplot(3, 2, ii + 1)
        ax = plt.imshow(mean_image.reshape(15,15), cmap='Greys')
        plt.title('Class: {}'.format(label))
    fig.savefig('figures/mean_images.pdf')

def table_meanstd_per_class(full_dataset, labels):
    for ii, label in enumerate(np.unique(labels)):
        ix = np.where(labels == label)
        mean_per_image = np.mean(full_dataset[list(ix[0]), :])
        std_per_image = np.std(full_dataset[list(ix[0]), :])

        print('mean label ', label, ': ', mean_per_image)
        print('std label ', label, ': ', std_per_image)
        print('max: ', np.max(full_dataset[list(ix[0]), :]))
        print('min: ', np.min(full_dataset[list(ix[0]), :]))

if __name__ == '__main__':
    seed = 8
    full_dataset, labels = load_data()
    full_dataset = np.array(full_dataset)

    #Hist by class
    full_dataset, labels = remove_black_images(full_dataset, labels)
    full_dataset = np.array(full_dataset)

    hist_by_class(full_dataset, labels)

    table_meanstd_per_class(full_dataset, labels)

    full_dataset, labels = normalize_ds(full_dataset, labels)

    dims = np.shape(full_dataset)
    print(dims)
    print(np.min(full_dataset), np.max(full_dataset))

    #fig, ax = plt.subplots(2,1) # Figure with 9 rows and 4 columns
    #sns.distplot(full_dataset[422], ax=ax[0], kde=True) # Use a single feature at a time
    #sns.distplot(full_dataset[4334], ax=ax[1], kde=True) # Use a single feature at a time
    #plt.show()

    #Mean image by class
    mean_image_by_class(full_dataset, labels)


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
    plt.title('Full dataset', format = 'pdf', bbox_inches = 'tight')

    for ii, label in enumerate(np.unique(labels)):
        ix = np.where(labels == label)
        mean_per_image = np.mean(full_dataset[list(ix[0]), :], axis = 1)
        plt.subplot(3, 2, ii + 2)
        ax = plt.hist(mean_per_image)
        plt.title('Class: {}'.format(label))
    fig.savefig('figures/hist_classes.pdf', format = 'pdf', bbox_inches = 'tight')


def mean_image_by_class(full_dataset, labels):
    #Mean image by class

    full_dataset = np.array(full_dataset)
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(9, 6))
    mean_image = np.mean(full_dataset, axis = 0)
    plt.subplot(3, 2, 1, aspect='equal')
    ax = plt.imshow(mean_image.reshape(15,15), cmap='Greys')
    plt.title('Full Dataset')
    plt.axis('off')
    for ii, label in enumerate(np.unique(labels)):
        ix = np.where(labels == label)
        mean_image = np.mean(full_dataset[list(ix[0]), :], axis = 0)
        plt.subplot(3, 2, ii + 2, aspect='equal')
        ax = plt.imshow(mean_image.reshape(15,15), cmap='Greys')
        plt.title('Class {}'.format(label))
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig('figures/mean_images.pdf', format = 'pdf', bbox_inches = 'tight')

def table_meanstd_per_class(full_dataset, labels):
    mean_per_image = np.mean(full_dataset)
    std_per_image = np.std(full_dataset)
    print('mean total: ', mean_per_image)
    print('std total: ', std_per_image)
    print('max: ', np.max(full_dataset))
    print('min: ', np.min(full_dataset))

    for ii, label in enumerate(np.unique(labels)):
        ix = np.where(labels == label)
        mean_per_image = np.mean(full_dataset[list(ix[0]), :])
        std_per_image = np.std(full_dataset[list(ix[0]), :])

        print('mean label ', label, ': ', mean_per_image)
        print('std label ', label, ': ', std_per_image)
        print('max: ', np.max(full_dataset[list(ix[0]), :]))
        print('min: ', np.min(full_dataset[list(ix[0]), :]))

def violin_plot(full_dataset, labels):
    mean_per_image = np.mean(full_dataset, axis = 1)
    print(mean_per_image.shape)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax = sns.violinplot(y=mean_per_image,
                        x=labels,
                        palette = ['navajowhite',
                                   'royalblue',
                                   'cornflowerblue' ,
                                   'lightsteelblue',
                                   'slategrey'])
    #ax.set_title('Distribution of image means by class')
    ax.set_ylabel('Brightness',fontsize=20)
    ax.set_xlabel('Image Class',fontsize=20)
    ax.tick_params(labelsize=15)
    fig.savefig('figures/violin.pdf', format = 'pdf', bbox_inches = 'tight')

if __name__ == '__main__':
    seed = 8
    full_dataset, labels = load_data()
    full_dataset = np.array(full_dataset)

    #Hist full
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    mean_per_image = np.mean(full_dataset, axis = 1)
    ax = sns.distplot(mean_per_image) #, bins = 50)
    ax.set_xlabel('Brightness')
    fig.savefig('figures/histogram.pdf', format = 'pdf', bbox_inches = 'tight')

    #Barplot Labels
    len_list = []
    fig, ax = plt.subplots(1, 1) #, figsize=(10, 10))
    for ll in np.unique(labels):
        labels_list = [label for label in labels if label == ll]
        len_list.append(len(labels_list))
    print(np.unique(labels), len_list)
    ax = sns.barplot(x=np.unique(labels),
                     y=len_list,
                     palette = ['navajowhite',
                                   'royalblue',
                                   'cornflowerblue' ,
                                   'lightsteelblue',
                                   'slategrey']) #, bins = 50)
    fig.savefig('figures/labelsdistr.pdf', format = 'pdf', bbox_inches = 'tight')


    #Hist by class
    full_dataset, labels = remove_black_images(full_dataset, labels)
    full_dataset = np.array(full_dataset)
    #hist_by_class(full_dataset, labels)

    #Table means per class
    #table_meanstd_per_class(full_dataset, labels)

    #Example Images
    #fig, ax = plt.subplots(2,1) # Figure with 9 rows and 4 columns
    #sns.distplot(full_dataset[422], ax=ax[0], kde=True) # Use a single feature at a time
    #sns.distplot(full_dataset[4334], ax=ax[1], kde=True) # Use a single feature at a time
    #plt.show()

    #Normalize Dataset
    #full_dataset, labels = normalize_ds(full_dataset, labels)

    #Mean image by class
    mean_image_by_class(full_dataset, labels)

    #Violin Plot
    violin_plot(full_dataset, labels)

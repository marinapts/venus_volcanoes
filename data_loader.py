from src.pyvov import ChipsIndex
from random import shuffle
import numpy as np
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, experiment_names=['C1', 'D4'], validation_ratio=0.1, testset_ratio=0.1, seed=8):
        ci = ChipsIndex()

        if experiment_names == ['C1', 'D4']:
            self.full_dataset, self.all_labels = ci.get_all()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.full_dataset, self.all_labels, test_size=testset_ratio, random_state=seed)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=validation_ratio, random_state=seed)

        else:
            self.full_dataset, self.all_labels = ci.get_specific(experiment_names)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.full_dataset, self.all_labels, test_size=testset_ratio, random_state=seed)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=validation_ratio, random_state=seed)

    def convert_to_numpy_sets(self, binary_class=False):
        """
        Returns the train, validation and test sets and labels as numpy arrays

        :param self: volcano data object that contains all the data as lists containing square images as
        flattened lists
        :param binary_class: boolean that determines whether we want to perform binary classification on
        class 0 versus all of 1, 2, 3, 4
        :return: (X_train, y_train), (X_val, y_val), (X_test, y_test): (input, output) numpy ndarray pairs
        of the format (n_samples, n_channels, n_rows, n_columns)
        """
        N_train = len(self.training_labels)
        N_val = len(self.validation_labels)
        N_test = len(self.testing_labels)

        img_len = len(self.testing_set[0])  # obtain length of flattened image
        num_rows = int(np.sqrt(img_len))
        num_cols = int(np.sqrt(img_len))
        num_channels = 1  # only have pixel intensities, no colour

        X_train = np.asarray(self.training_set).reshape((N_train, num_channels, num_rows, num_cols))
        X_val = np.asarray(self.validation_set).reshape((N_val, num_channels, num_rows, num_cols))
        X_test = np.asarray(self.testing_set).reshape((N_test, num_channels, num_rows, num_cols))

        y_train = np.asarray(self.training_labels)
        y_val = np.asarray(self.validation_labels)
        y_test = np.asarray(self.testing_labels)

        if binary_class is True:
            np.place(y_train, mask=y_train > 0, vals=1)
            np.place(y_val, mask=y_val > 0, vals=1)
            np.place(y_test, mask=y_test > 0, vals=1)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_training_set(self):
        return self.X_train, self.y_train

    def get_validation_set(self):
        return self.X_val, self.y_val

    def get_testing_set(self):
        return self.X_test, self.y_test

    def get_full_dataset(self):
        return self.full_dataset, self.all_labels

    def get_data_tuple(self):
        """
        Acts like a getter, but returns all sets at once as a tuple
        :return: tuple of all splits and corresponding labels as (set, labels), (set, labels), ...
        """
        return self.X_train, self.y_train, self.X_val, self.y_val, \
               self.X_test, self.y_test


if __name__ == "__main__":
    data = DataLoader()
    train, labels = data.get_training_set()
    val, labels = data.get_validation_set()
    test, labels = data.get_testing_set()
    full, labels = data.get_full_dataset()
    # print(len(data.get_training_set()), len(data.get_validation_set()), len(data.get_testing_set()))
    print(len(train), len(val), len(test))
    print(len(full))

    data = DataLoader('A1')

    train_a1 = data.get_training_set()
    print(len(train_a1))

from src.pyvov import ChipsIndex
from random import shuffle
import numpy as np


class DataLoader:
    def __init__(self, experiment_names=['C1', 'D4'], val_ratio=0.1, test_ratio=0.1, seed=8):
        ci = ChipsIndex()

        # all_experiments = ci.experiments()
        # EXP_NAMES = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'C1', 'D1', 'D2', 'D3', 'D4',
        # 'E1', 'E2', 'E3', 'E4', 'E5']
        # num_img = 0

        training_split = []
        testing_split = []
        all_labels = []

        # Obtain experiment data, combine C1 and D4 training and test sets:
        for EXP_NAME in experiment_names:
            training_split.extend(ci.training_split_for(EXP_NAME))
            testing_split.extend(ci.testing_split_for(EXP_NAME))
            labels = ci.labels_for(EXP_NAME)
            label_list = list(labels['trn'])
            label_list.extend(list(labels['tst']))
            all_labels.extend(label_list)

        training_split.extend(testing_split)
        full_dataset = training_split

        # Shuffle the data
        ordering = np.arange(len(full_dataset))
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(ordering)
        full_dataset = [full_dataset[i] for i in ordering]
        self.all_labels = [all_labels[i] for i in ordering]

        # Create training, validation and test sets
        self.validation_set = full_dataset[0:int(val_ratio * (len(full_dataset)))]
        self.validation_labels = self.all_labels[0:int(val_ratio * (len(full_dataset)))]

        self.testing_set = full_dataset[
                           int(val_ratio * (len(full_dataset))):int((val_ratio + test_ratio) * (len(full_dataset)))]
        self.testing_labels = self.all_labels[
                              int(val_ratio * (len(full_dataset))):int((val_ratio + test_ratio) * (len(full_dataset)))]

        self.training_set = full_dataset[int((val_ratio + test_ratio) * (len(full_dataset))):]
        self.training_labels = self.all_labels[
                               int((val_ratio + test_ratio) * (len(full_dataset))):]

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
        return self.training_set

    def get_training_labels(self):
        return self.training_labels

    def get_validation_set(self):
        return self.validation_set

    def get_validation_labels(self):
        return self.validation_labels

    def get_testing_set(self):
        return self.testing_set

    def get_testing_labels(self):
        return self.testing_labels

    def get_full_dataset(self):
        return self.training_set + self.validation_set + self.testing_set

    def get_all_labels(self):
        return self.all_labels

    def get_data_tuples(self):
        """
        Acts like a getter, but returns all sets at once as tuples for one-line
        :return: tuple of (data, labels)
        """
        return self.training_set, self.training_labels, self.validation_set, self.validation_labels, \
               self.testing_set, self.testing_labels


if __name__ == "__main__":
    data = DataLoader()
    train = data.get_training_set()
    val = data.get_validation_set()
    test = data.get_testing_set()
    full = data.get_full_dataset()
    labels = data.get_all_labels()
    # print(len(data.get_training_set()), len(data.get_validation_set()), len(data.get_testing_set()))
    print(len(train), len(val), len(test))
    print(len(full))

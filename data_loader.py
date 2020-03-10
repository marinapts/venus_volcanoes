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

        # Combine C1 and D4 training and test sets:
        training_split = []
        testing_split = []
        all_labels = []

        for EXP_NAME in experiment_names:
            training_split.extend(ci.training_split_for(EXP_NAME))
            testing_split.extend(ci.testing_split_for(EXP_NAME))
            labels = ci.labels_for(EXP_NAME)
            label_list = list(labels['trn'])
            label_list.extend(list(labels['tst']))
            all_labels.extend(label_list)

        training_split.extend(testing_split)
        full_dataset = training_split

        ordering = np.arange(len(full_dataset))

        # Create Training, validation and test sets
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(ordering)
        full_dataset = [full_dataset[i] for i in ordering]
        self.all_labels = [all_labels[i] for i in ordering]

        # shuffle(full_dataset)
        self.validation_set = full_dataset[0:int(val_ratio * (len(full_dataset)))]
        self.testing_set = full_dataset[
                           int(val_ratio * (len(full_dataset))):int((val_ratio + test_ratio) * (len(full_dataset)))]
        self.training_set = full_dataset[int((val_ratio + test_ratio) * (len(full_dataset))):]

    def get_training_set(self):
        return self.training_set

    def get_validation_set(self):
        return self.validation_set

    def get_testing_set(self):
        return self.testing_set

    def get_full_dataset(self):
        return self.training_set + self.validation_set + self.testing_set

    def get_labels(self):
        return self.all_labels


if __name__ == "__main__":
    data = DataLoader()
    train = data.get_training_set()
    val = data.get_validation_set()
    test = data.get_testing_set()
    full = data.get_full_dataset()
    labels = data.get_labels()
    # print(len(data.get_training_set()), len(data.get_validation_set()), len(data.get_testing_set()))
    print(len(train), len(val), len(test))
    print(len(full))


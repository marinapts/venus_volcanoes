from data_loader import DataLoader
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN


def reshape_X(X_train, before_augmentation=True):
    print('Original dataset', X_train.shape)

    if before_augmentation is True:
        num_images, _, img_h, img_w = X_train.shape
        # Before any augmentation is applied
        X_reshaped = X_train.reshape(num_images, -1)
    else:
        # Reshape back to original shape
        num_augm_images = X_train.shape[0]
        X_reshaped = np.asarray(X_train).reshape(num_augm_images, 1, 15, 15)
        print(X_reshaped.shape)

    return X_reshaped


def smote(X_train, y_train):
    X_reshaped = reshape_X(X_train, before_augmentation=True)

    print('Apply SMOTE')
    sm = SMOTE(random_state=42)
    X_smote, y_smote = sm.fit_resample(X_reshaped, y_train)
    smote_counter = Counter(y_smote)
    print(smote_counter)

    X_smote = reshape_X(X_smote, before_augmentation=False)

    return X_smote, np.asarray(y_smote)


def adasyn(X_train, y_train):
    X_reshaped = reshape_X(X_train, before_augmentation=True)

    counter = Counter(y_train)
    print('Original dataset', counter)

    ada = ADASYN(random_state=42)
    X_res, y_res = ada.fit_resample(X_reshaped, y_train)
    print('Resampled dataset', Counter(y_res))

    X_res = reshape_X(X_res, before_augmentation=False)

    return X_res, np.asarray(y_res)


if __name__ == '__main__':
    data = DataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = data.convert_to_numpy_sets(binary_class=True)
    counter = Counter(y_train)

    print('X_train:', len(X_train))
    print(X_train[0].reshape(225).shape)
    print(counter)

    smote(X_train, y_train)

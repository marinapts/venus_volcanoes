from data_loader import DataLoader
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


def smote(X_train, y_train):
    print('initial', X_train.shape)
    num_images, _, img_h, img_w = X_train.shape
    X_reshaped = X_train.reshape(num_images, -1)

    print('Apply SMOTE')
    sm = SMOTE(random_state=42)
    X_smote, y_smote = sm.fit_resample(X_reshaped, y_train)
    smote_counter = Counter(y_smote)
    print(smote_counter)

    # Reshape back to original shape
    num_augm_images = X_smote.shape[0]
    X_smote = np.asarray(X_smote).reshape(num_augm_images, _, img_h, img_w)

    return X_smote, np.asarray(y_smote)


if __name__ == '__main__':
    data = DataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = data.convert_to_numpy_sets(binary_class=True)
    counter = Counter(y_train)

    print('X_train:', len(X_train))
    print(X_train[0].reshape(225).shape)
    print(counter)

    smote(X_train, y_train)

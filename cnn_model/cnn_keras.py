import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns

import data_loader as volcano_data_loader

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils

from sklearn.metrics import classification_report
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix
from keras.callbacks.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.callbacks.callbacks import ModelCheckpoint


def plot_confusion_matrix(labels, predictions, p=0.5):
    """
    Implements a confusion matrix via a seaborn heatmap

    :param labels: true labels, numpy array of shape (n_samples,)
    :param predictions: predicted labels, numpy array of shape (n_samples,)
    :param p: probability threshold for binary classification
    """
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')


def preprocess_data(X_train, y_train, X_val, y_val, X_test, y_test, binary=False):
    """
    Preprocesses the data by:
        - ensuring elements are of type 32 bit floats
        - transforming the range of pixel intensity values to [0, 1]
        - one-hot encoding the labels

    :param X_train: train input, numpy array of shape (n_samples, n_channels, n_rows, n_cols)
    :param y_train: train labels, numpy array of shape (n_samples,)
    :param X_val: val input, numpy array of shape (n_samples, n_channels, n_rows, n_cols)
    :param y_val: val labels, numpy array of shape (n_samples,)
    :param X_test: test input, numpy array of shape (n_samples, n_channels, n_rows, n_cols)
    :param y_test: test labels, numpy array of shape (n_samples,)
    :return: preprocessed data tuples (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
    """
    # Ensure that arrays have the correct data type
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')

    # todo: normalise the input data differently, if necessary
    # Transform the input data from [0, 255] to [0, 1]
    X_train /= 255
    X_val /= 255
    X_test /= 255

    num_classes = np.unique(y_train).shape[0]

    # One-hot encode the labels if performing multi-label classification
    if binary is True:
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    else:
        Y_train = np_utils.to_categorical(y_train, num_classes)
        Y_val = np_utils.to_categorical(y_val, num_classes)
        Y_test = np_utils.to_categorical(y_test, num_classes)
        return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def make_model(input_rows=15, input_cols=15, num_classes=5):
    """
    Creates an instance of a Keras model class with the CNN architecture used

    :param input_rows: number of rows of the input images
    :param input_cols: number of columns of the input images
    :param num_classes: number of output classes for classification (size of the one-hot vectors)
    :return: model
    """
    # TODO: decide on final model architecture

    # Define the model
    model = Sequential()

    # model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, num_rows, num_cols), padding='same'))
    # model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Convolution2D(8, (3, 3), activation='relu',
                            input_shape=(1, input_rows, input_cols),
                            padding='same',
                            data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', data_format='channels_first'))

    # model.add(Dropout(0.25))

    model.add(Convolution2D(16, (4, 4), activation='relu', padding='valid', data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', data_format='channels_first'))

    # model.add(Dropout(0.25))
    #
    # model.add(Convolution2D(32, (4, 4), activation='relu', padding='valid', data_format='channels_first'))
    # model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', data_format='channels_first'))

    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))

    # model.add(Dropout(0.5))

    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))

    return model


def create_class_weight_dict(y_train):
    """
    Creates a dictionary for class weights according to training data

    :param y_train: training labels, numpy array of size (n_samples,)
    :param binary: boolean determining whether we perform binary or multilabel classification
    :return: dictionary of class weights
    """

    class_weight_vector = compute_class_weight('balanced', np.unique(y_train), y_train)

    class_weight_dict = dict()
    classes = np.unique(y_train)
    for i in classes:
        class_weight_dict[i] = class_weight_vector[i]

    return class_weight_dict


def train_model(model, X_train, Y_train, X_val, Y_val, exp_checkpoint_dir, num_epochs=100, batch_size=32,
                early_stopping_patience=0, class_weight_dict=None):
    """
    Trains a given Keras model according to user specifications

    :param model: Keras model object
    :param X_train: train input, numpy array of shape (n_samples, n_channels, n_rows, n_cols)
    :param y_train: train labels, numpy array of shape (n_samples,)
    :param X_val: val input, numpy array of shape (n_samples, n_channels, n_rows, n_cols)
    :param y_val: val labels, numpy array of shape (n_samples,)
    :param exp_checkpoint_dir: absolute directory for storing the model weights
    :param num_epochs: number of epochs to run, default 32
    :param batch_size: batch size, default 32
    :param early_stopping_patience: number of epochs to run without improvement in validation loss before the
                                    model stops
    :param class_weight_dict: dictionary of class labels and class weights as key:value pairs for training
    """

    # If early stopping patience is zero, don't use early stopping,
    # but we still want to use the weights from the best model (using patience=0 seems to have a bug)
    if early_stopping_patience == 0:
        patience = num_epochs
    else:
        patience = early_stopping_patience

    early_stopping_monitor = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=patience,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )
    weights_path = os.path.join(exp_checkpoint_dir, 'weights.hdf5')

    mcp_save = ModelCheckpoint(weights_path,
                               save_best_only=True, monitor='val_loss', mode='min')

    model.fit(X_train, Y_train,
              validation_data=(X_val, Y_val),
              batch_size=batch_size,
              epochs=num_epochs,
              verbose=2,
              class_weight=class_weight_dict,
              callbacks=[early_stopping_monitor, mcp_save])


def main():
    # TODO (optional): parse constants from arguments
    EXPERIMENT_NAME = 'experimental'
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3  # Default for Adam is 1e-3
    USE_BINARY_CLASS = True
    CLASS_WEIGHT_BALANCING = True

    # Set seed for consistent results
    np.random.seed(8)

    # Make directory for experiment
    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
    exp_checkpoint_dir = os.path.join(checkpoint_dir, EXPERIMENT_NAME)
    load_path = os.path.join(exp_checkpoint_dir, 'weights.hdf5')
    try:
        os.makedirs(exp_checkpoint_dir)
    except FileExistsError:
        pass
    else:
        raise Exception('Something else went wrong when creating the directory')

    # Load volcano data object and the corresponding train, val and test sets
    print('Loading volcano data')
    volcano_data = volcano_data_loader.DataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = volcano_data.convert_to_numpy_sets(
        binary_class=USE_BINARY_CLASS)

    print('Pre-processing data')
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = preprocess_data(X_train, y_train,
                                                                           X_val, y_val,
                                                                           X_test, y_test, binary=USE_BINARY_CLASS)

    num_rows = X_train.shape[2]
    num_cols = X_train.shape[3]
    num_classes = np.unique(y_train).shape[0]

    # Todo: check other ways of dealing with class imbalance
    # Data is highly imbalanced, use class weights for training
    if CLASS_WEIGHT_BALANCING is True:
        class_weight_dict = create_class_weight_dict(y_train)
        print('Using class_weight_dict ', class_weight_dict)
    else:
        class_weight_dict = None
        print('Not using class weight balancing')

    model = make_model(input_rows=num_rows, input_cols=num_cols, num_classes=num_classes)

    # Compile the model
    if USE_BINARY_CLASS is True:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'

    model.compile(loss=loss,
                  optimizer=Adam(learning_rate=LEARNING_RATE),
                  metrics=['accuracy'])

    print('Constructed model:')
    print(model.summary())

    # Training
    train_model(model, X_train, Y_train, X_val, Y_val, exp_checkpoint_dir=exp_checkpoint_dir, num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE, early_stopping_patience=0, class_weight_dict=class_weight_dict)

    print('Loading best model from checkpoints...')
    model.load_weights(load_path)

    # Evaluation of model with best validation performance
    loss, acc = model.evaluate(X_val, y_val, verbose=2)
    print('Showing performance summary of the best model')
    print("Accuracy: {:5.2f}%".format(100 * acc))
    print("Loss: {:10.9f}".format(loss))

    # Get evaluation metrics (overall loss and accuracy for now)
    # score = model.evaluate(X_val, Y_val, verbose=0)
    # print('Overall evaluation metric: ', score)

    y_val_pred = model.predict_classes(X_val)
    print(classification_report(y_val, y_val_pred))
    plot_confusion_matrix(y_val, y_val_pred)
    plt.show()


if __name__ == '__main__':
    main()

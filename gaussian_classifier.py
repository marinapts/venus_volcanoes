from data_loader import DataLoader
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import numpy as np
from src.pyvov import ChipsIndex


class gaussian_clf():
    def __init__(self, threshold=0.5, normalize=False):
        self.threshold=threshold
        self.normalize=normalize
    
    def train(self, X_train, y_train):
        
        if self.normalize:
            X_train = normalize(X_train)
        
        self.pca = principal_components(X_train, y_train, 35)
        X_train = self.pca.transform(X_train)

        new_labels = np.zeros_like(y_train)
        for idx, item in enumerate(y_train):
            if item > 0:
                new_labels[idx] += 1

        label = new_labels
        
        positives = X_train[np.where(y_train>0)]
        negatives = X_train[np.where(y_train==0)]
        assert positives.shape[0]+negatives.shape[0]==X_train.shape[0]
        
        self.mean_0 = negatives.mean(axis=0)
        self.cov_0 = np.cov(negatives.T)
        self.mean_1 = positives.mean(axis=0)
        self.cov_1 = np.cov(positives.T)
        
        self.prior = positives.shape[0] / (positives.shape[0] + negatives.shape[0])
    
    def predict_prob(self, X):
        self.score_0 = multivariate_normal.pdf(X, mean=self.mean_0, cov=self.cov_0)
        self.score_1 = multivariate_normal.pdf(X, mean=self.mean_1, cov=self.cov_1)
        posterior = self.score_1 * self.prior / (self.score_1 * self.prior + self.score_0 * (1-self.prior))
        
        return posterior
        
    def evaluate(self, X, y):
        if self.normalize:
            X = normalize(X)
        
        X = self.pca.transform(X)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
        for idx in range(X.shape[0]):
            if self.predict_prob(X[idx]) > self.threshold:
                if y[idx]==1:
                    tp += 1
                else:
                    fp += 1
            
            else:
                if y[idx]==1:
                    fn += 1
                else:
                    tn += 1
        
        # print('true_positives:', tp)
        # print('true_negatives:', tn)
        # print("false_positives:", fp)
        # print("false_negatives:", fn, "\n")
        print("recall:", tp/(tp+fn))
        print("precision:", tp/(tp+fp))

        recall = tp/(tp+fn)
        precision = tp/(tp+fp)

        f1 = 2 * precision * recall / (precision + recall)

        print("F1 score:", f1)

def get_sets():
    """
    This takes ALL of the available datapoints and makes train/val/test splits.
    Needs to be improved upon by looking at specific experiments e.g HOM36 etc.
    """
    
    data = DataLoader()
    X_train = data.get_training_set()
    X_train = np.asarray(X_train)
    
    X_val = data.get_validation_set()
    X_val = np.asarray(X_val)
    
    X_test = data.get_testing_set()
    X_test = np.asarray(X_test)
    
    labels = data.get_labels()
    labels = np.asarray(labels)

    
    n = new_labels.shape[0]
    y_val = new_labels[0:int(0.1*n)]
    y_test = new_labels[int(0.1*n):int(0.2*n)]
    y_train = new_labels[int(0.2*n):]
    
    print('X_train shape:', X_train.shape, '   y_train shape:', y_train.shape)
    print('X_val shape:', X_val.shape, '   y_val shape:', y_val.shape)
    print('X_test shape:', X_test.shape, '   y_test shape:', y_test.shape)
    
    return X_train, y_train, X_val, y_val, X_test, y_test
            
def principal_components(X, y, n):
    
    positives = X[np.where(y>0)]
    
    pca = PCA(n_components=n)
    pcs = pca.fit(positives)
    # print("Singular values:",pca.singular_values_)
    # print("Explained variance:", pca.explained_variance_)
    
    return pca

def normalize(X):
    X = X.astype(float)
    new_X = np.zeros_like(X)
    
    for idx in range(X.shape[0]):
        # new_X[idx] = X[idx]
        new_X[idx] = X[idx] - np.mean(X[idx])
        new_X[idx] = new_X[idx]/np.std(X[idx])
        
    return new_X

def evaluate_experiment(exp_name='with all data'):

    print("evaluating experiment {}".format(exp_name))

    data = DataLoader()
    
    X_train, y_train = data.get_training_set()
    X_val, y_val = data.get_validation_set()
    X_test, y_test = data.get_testing_set()

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    ## clean data (remove all-zero datapoints)
    removed_data = 0
    m = np.all(X_train[:, 1:] == X_train[:, :-1], axis=1)*1
    mask = np.where(m>0)
    X_train = np.delete(X_train, mask, axis=0)
    y_train = np.delete(y_train, mask, axis=0)
    removed_data += np.sum(m)
    m = np.all(X_test[:, 1:] == X_test[:, :-1], axis=1)*1
    mask = np.where(m>0)
    X_test = np.delete(X_test, mask, axis=0)
    y_test = np.delete(y_test, mask, axis=0)
    removed_data += np.sum(m)
    m = np.all(X_val[:, 1:] == X_val[:, :-1], axis=1)*1
    mask = np.where(m>0)
    X_val = np.delete(X_val, mask, axis=0)
    y_val = np.delete(y_val, mask, axis=0)
    removed_data += np.sum(m)
    print("Number of all-zero entries removed:", removed_data)


    y_train[np.where(y_train>0)]=1
    y_val[np.where(y_val>0)]=1
    y_test[np.where(y_test>0)]=1
    
    """
    set thresholds for evaluation as you like -- these are referred to as
    operating points in the paper.
    """
    baseline = gaussian_clf(threshold=0.5, normalize=True)
    baseline.train(X_train, y_train)
    baseline.evaluate(X_val, y_val)


if __name__ == "__main__":
	evaluate_experiment()







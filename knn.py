import numpy as np
import pandas as pd
from collections import Counter


def k_fold_cross_validation(k, data):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds_indices = []
    for fold_i in range(k):
        test_indices = indices[fold_i * fold_size: (fold_i + 1) * fold_size]
        train_indices = np.concatenate([indices[:fold_i * fold_size], indices[(fold_i + 1) * fold_size:]])
        folds_indices.append((train_indices, test_indices))

    return folds_indices


def knn(datas, f_indices, k):
    features = np.array(datas.iloc[:, :-1].values)
    labels = np.array(datas.iloc[:, -1].values)

    count_correct = 0
    for train_index, test_index in f_indices:
        x_train, y_train = features[train_index], labels[train_index]
        x_test, y_test = features[test_index], labels[test_index]

        # feature_diffs in size (1-fold) * (k-1)fold * feature_size
        feature_diffs = x_test[:, None, :] - x_train[None, :, :]
        distance = np.sum(feature_diffs ** 2, axis=2)   # Square of all feature diffs
        k_near_indices = np.argsort(distance, axis=1)[:, :k]    
       
        predicted_labels = [Counter(y_train[k_near].flat).most_common(1)[0][0] for k_near in k_near_indices]
        fold_correct = np.array([predicted_labels == y_test]).sum()
        count_correct += fold_correct

    return (count_correct / len(datas)) * 100


if __name__ == '__main__':
    dataset = pd.read_csv('IRIS.csv')
    df = dataset.sample(frac=1).reset_index(drop=True)   # Shuffle data
    folds = k_fold_cross_validation(10, df)
    print("Accuracy is: {} percent".format(knn(df, folds, 10), '.2f'))

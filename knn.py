import numpy as np
import pandas as pd
from collections import Counter


def manhattan_distance(x_test_f, x_train_f):
    # feature_diffs in a 3D array with size: (1-fold-size) * (k-1)-fold-size * feature-size
    feature_diffs = x_test_f[:, None, :] - x_train_f[None, :, :]
    distances = np.sum(abs(feature_diffs), axis=2)   # sum them up and get square root
    return distances


def euclidean_distance(x_test_f, x_train_f):
    # feature_diffs in a 3D array with size: (1-fold-size) * (k-1)-fold-size * feature-size
    feature_diffs = x_test_f[:, None, :] - x_train_f[None, :, :]
    # Power 2 of all feature diffs and sum them up and get square root
    distances = np.sqrt(np.sum(feature_diffs ** 2, axis=2))
    return distances


def k_fold_cross_validation(k, data):
    fold_size = len(data) // k
    indices = np.arange(len(data))  # data row indices
    folds_indices = []  # each element consists of 2 arrays : train indices + test indices
    for fold_i in range(k):
        # Kth first fold-size indices are test indices
        test_indices = indices[fold_i * fold_size: (fold_i + 1) * fold_size]
        train_indices = np.concatenate([indices[:fold_i * fold_size], indices[(fold_i + 1) * fold_size:]])
        folds_indices.append((train_indices, test_indices))

    return folds_indices


def knn(datas, f_indices, k):
    fold_k = len(f_indices[0][1])
    features = np.array(datas.iloc[:, :-1].values)
    labels = np.array(datas.iloc[:, -1].values)

    max_correct = 0
    count_correct = 0
    for train_index, test_index in f_indices:
        x_train, y_train = features[train_index], labels[train_index]
        x_test, y_test = features[test_index], labels[test_index]

        distance_matrix = manhattan_distance(x_test, x_train)
        k_near_indices = np.argsort(distance_matrix, axis=1)[:, :k]

        predicted_labels = [Counter(y_train[k_near].flat).most_common(1)[0][0] for k_near in k_near_indices]
        fold_correct = np.array([predicted_labels == y_test]).sum()
        count_correct += fold_correct

        if fold_correct > max_correct:
            max_correct = fold_correct

    max_accuracy = (max_correct / fold_k) * 100
    mean_accuracy = (count_correct / len(datas)) * 100
    return mean_accuracy, max_accuracy


if __name__ == '__main__':
    df = pd.read_csv('IRIS.csv')
    df = df.sample(frac=1).reset_index(drop=True)

    folds = k_fold_cross_validation(10, df)
    mean_acc, max_acc = knn(df, folds, 3)
    print("Mean Accuracy = {:.2f} percent \nMax Accuracy = {:.2f} percent".format(mean_acc, max_acc))

import numpy as np
import pandas as pd
from collections import Counter


def manhattan_distance(x_test_f, x_train_f):
    # feature_diffs in a 3D array with size: test-size * train-size * feature-size matrix
    feature_diffs = x_test_f[:, None, :] - x_train_f[None, :, :]
    distances = np.sum(abs(feature_diffs), axis=2)   # sum them up and get square root
    return distances


def euclidean_distance(x_test_f, x_train_f):
    # feature_diffs in a 3D array with size: test-size * train-size * feature-size matrix
    feature_diffs = x_test_f[:, None, :] - x_train_f[None, :, :]
    # Power 2 of all feature diffs and sum them up and get square root
    distances = np.sqrt(np.sum(feature_diffs ** 2, axis=2))
    return distances


def knn(x_tr, y_tr, x_te, y_te, k_n):
    fold_k = len(y_te)
    distance_matrix = euclidean_distance(x_te, x_tr)    # test-size * train-size * feature-size matrix
    k_near_indices = np.argsort(distance_matrix, axis=1)[:, :k_n]
    predicted_labels = [Counter(y_tr[k_near].flat).most_common(1)[0][0] for k_near in k_near_indices]
    fold_correct = np.array([predicted_labels == y_te]).sum()
    return (fold_correct / fold_k) * 100


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


def k_fold_knn(datas, f_indices, k):
    print('#' * 10, 'K-Fold KNN', '#' * 10)
    print('Fold size = {}, Number of folds = {}, K for KNN = {}'
          .format(len(f_indices[0][1]), len(f_indices), k))
    print('Fold Accuracies : ', end='')
    features = np.array(datas.iloc[:, :-1].values)
    labels = np.array(datas.iloc[:, -1].values)

    max_accuracy = 0
    mean_accuracy = 0
    for train_index, test_index in f_indices:
        x_train, y_train = features[train_index], labels[train_index]   # fold train set
        x_test, y_test = features[test_index], labels[test_index]   # fold test set

        accuracy = knn(x_train, y_train, x_test, y_test, k)     # apply knn
        print('{:.2f}'.format(accuracy), end=', ')
        max_accuracy = max(accuracy, max_accuracy)
        mean_accuracy += accuracy

    mean_accuracy = (mean_accuracy / len(f_indices))
    return mean_accuracy, max_accuracy


if __name__ == '__main__':
    df = pd.read_csv('IRIS.csv')
    # df = df.sample(frac=1).reset_index(drop=True)   # uncomment this if you want to add data shuffling

    best = [0, 0, 0, 0]
    for i in range(5, 25, 5):
        for j in range(3, 11):
            folds = k_fold_cross_validation(i, df)
            mean_acc, max_acc = k_fold_knn(df, folds, j)
            print("\nMean Accuracy = {:.2f} percent \nMax Accuracy = {:.2f} percent\n"
                  .format(mean_acc, max_acc))
            if mean_acc > best[0]:
                best = [mean_acc, max_acc, i, j]

    print("\nBest Mean Accuracy = {:.2f}\nBest Max Accuracy = {:.2f}"
          .format(best[0], best[1]))
    print("\nBest K-fold Size = {}\nBest K for KNN = {}"
          .format(best[2], best[3]))

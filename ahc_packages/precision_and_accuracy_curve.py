import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt


def precision_accuracy_curve(X, y_true, w, b, num_thresholds=100, save_path=''):
    # calculate prediction score
    scores = np.dot(X, w) + b

    # set thresholds
    start_threshold = np.max(scores)
    stop_threshold = np.min(scores) - 1
    thresholds = np.linspace(start_threshold, stop_threshold, num_thresholds)

    precisions = []
    accuracies = []

    for threshold in thresholds:
        y_pred = np.where(scores >= threshold, 1, 0)
        precision = precision_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precisions.append(precision)
        accuracies.append(accuracy)

    plt.figure()
    plt.plot(precisions, accuracies, marker='o', linestyle='-')
    plt.xlabel('Precision')
    plt.ylabel('Accuracy')
    plt.title('Precision-Accuracy curve')
    plt.grid(True)
    plt.xlim(0.35, 1.05)
    plt.ylim(0.35, 1.05)
    plt.savefig(save_path)
    plt.close('all')
    return accuracies, precisions, thresholds



# file_path = 'rice'
# n_iters = 1000
# results = train_and_evaluate(file_path=file_path, n_iters=n_iters, positive_size=2500/3810, negative_size=2500/3810)
# print(results['weights'])
# print(np.sum(np.abs(results['weights'])))
# print(results['train_recall'])
# print(results['train_precision'])
# print(results['train_accuracy'])
# print(results['test_recall'])
# print(results['test_precision'])
# print(results['test_accuracy'])
# X_train, y_train = results['X_train'], results['y_train']
# weights = results['weights']
# bias = results['bias']
# accuracies, precisions, thresholds = precision_accuracy_curve(X_train, y_train, weights, bias)
# # draw precision-accuracy curve

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import adult_sample_data
import bank_sample_data
import adult_processed_sample_data
from sklearn.linear_model import LogisticRegression
import rice_sample_data
import churn_sample_data


class LinearClassifier:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.w) + self.b
            y_pred = self._sigmoid(linear_model)

            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.b
        y_pred = self._sigmoid(linear_model)
        return np.round(y_pred)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


def train_and_evaluate(file_path, n_iters=1000, positive_size=1.0, negative_size=1.0):
    # Load and preprocess data
    # X, y = load_and_preprocess_data(file_path)
    if file_path == 'C:/Users/zhengke/Desktop/24_07/20240704/BinaryClassification/adult/adult_processed.csv':
        X, y = adult_processed_sample_data.sample_data(file_path=file_path, random_seed=42, positive_size=positive_size,
                                                       negative_size=negative_size)
    elif file_path == 'rice':
        X, y = rice_sample_data.sample_data(file_path=None, random_seed=42, positive_size=positive_size,
                                            negative_size=negative_size)
    elif file_path == 'churn':
        X, y = churn_sample_data.sample_data(file_path=None, random_seed=42, positive_size=positive_size,
                                             negative_size=negative_size)
    elif file_path == 'C:/Users/zhengke/Desktop/24_07/20240704/BinaryClassification/bank+marketing/bank/bank-full.csv':
        X, y = bank_sample_data.sample_data(file_path=file_path, random_seed=42, positive_size=positive_size,
                                            negative_size=negative_size)
    else:
        X = None
        y = None
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=41)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train linear classifier
    # clf = LinearClassifier(learning_rate=learning_rate, n_iters=n_iters)
    # clf.fit(X_train, y_train)
    clf = LogisticRegression(max_iter=n_iters, C=1)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Calculate metrics
    metrics = {}

    metrics['train_accuracy'] = accuracy_score(y_train, y_pred_train)
    metrics['test_accuracy'] = accuracy_score(y_test, y_pred_test)

    metrics['train_precision'] = precision_score(y_train, y_pred_train)
    metrics['train_recall'] = recall_score(y_train, y_pred_train)

    metrics['test_precision'] = precision_score(y_test, y_pred_test)
    metrics['test_recall'] = recall_score(y_test, y_pred_test)

    # metrics['weights'] = clf.w
    # metrics['bias'] = clf.b
    metrics['weights'] = clf.coef_[0]
    metrics['bias'] = clf.intercept_[0]

    metrics[
        'X_train'] = X_train  # return X_train generated randomly by train_test_split with specific random_state(seed)
    metrics['y_train'] = y_train
    metrics['X_test'] = X_test  # return X_train generated randomly by train_test_split with specific random_state(seed)
    metrics['y_test'] = y_test
    return metrics  # Not only metric in fact


def precision_accuracy_curve(X, y_true, w, b, num_thresholds=100, save_path=''):
    # 计算预测分数
    scores = np.dot(X, w) + b

    # 设置阈值范围
    start_threshold = np.max(scores)
    stop_threshold = np.min(scores)-1
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


def evaluate_classification(X, y, weight, bias):  # X, y, w obtained by PIP, b obtained by PIP
    # Calculate predictions
    scores = np.dot(X, weight) + bias
    predictions = np.zeros_like(y)  # Initialize predictions array
    buffered_predictions = np.zeros_like(y)
    # Assign predictions based on decision boundary
    predictions[scores >= 0] = 1  # Predict 1 if weight^T X + bias >= 0
    predictions[scores < 0] = -1  # Predict -1 if weight^T X + bias < 0
    buffered_predictions[scores >= -1e-5] = 1  # Predict 1 if weight^T X + bias >= 0
    buffered_predictions[scores < -1e-5] = -1  # Predict -1 if weight^T X + bias < 0
    # Convert y from -1/1 to 0/1 for compatibility with predictions
    y_binary = (y + 1) // 2  # 1 if y is 1, 0 if y is -1
    # Calculate metrics
    TP = np.sum((predictions == 1) & (y_binary == 1))
    FP = np.sum((predictions == 1) & (y_binary == 0))
    TN = np.sum((predictions == -1) & (y_binary == 0))
    FN = np.sum((predictions == -1) & (y_binary == 1))
    TP_ = np.sum((buffered_predictions == 1) & (y_binary == 1))
    FP_ = np.sum((buffered_predictions == 1) & (y_binary == 0))
    TN_ = np.sum((buffered_predictions == -1) & (y_binary == 0))
    FN_ = np.sum((buffered_predictions == -1) & (y_binary == 1))
    # Calculate precision, recall, and accuracy
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy = (TP + TN) / len(y)
    buffered_precision = TP_ / (TP_ + FP_) if (TP_ + FP_) > 0 else 0.0
    buffered_recall = TP_ / (TP_ + FN_) if (TP_ + FN_) > 0 else 0.0
    buffered_accuracy = (TP_ + TN_) / len(y)
    real_results = {'recall': recall, 'precision': precision, 'accuracy': accuracy}
    buffered_results = {'recall': buffered_recall, 'precision': buffered_precision, 'accuracy': buffered_accuracy}
    return real_results, buffered_results


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

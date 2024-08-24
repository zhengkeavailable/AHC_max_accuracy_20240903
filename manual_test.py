import numpy as np


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

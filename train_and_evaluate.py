from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sample_data import bank_sample_data
from sample_data import adult_processed_sample_data
from sklearn.linear_model import LogisticRegression
from sample_data import rice_sample_data
from sample_data import churn_sample_data
import numpy as np


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
    clf = LogisticRegression(max_iter=n_iters, C=1)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Calculate metrics and store split results
    results = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test,
               'train_accuracy': accuracy_score(y_train, y_pred_train),
               'test_accuracy': accuracy_score(y_test, y_pred_test),
               'train_precision': precision_score(y_train, y_pred_train),
               'train_recall': recall_score(y_train, y_pred_train),
               'test_precision': precision_score(y_test, y_pred_test), 'test_recall': recall_score(y_test, y_pred_test),
               'weights': clf.coef_[0], 'bias': clf.intercept_[0]}

    return results


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


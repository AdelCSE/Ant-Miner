import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    hamming_loss as hl,)

from sklearn.metrics import (
    label_ranking_loss,
    coverage_error,
    average_precision_score,)

# Accuracy 
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    accs = []
    for i in range(y_true.shape[1]):
        accs.append(accuracy_score(y_true[:, i], y_pred[:, i]))
    return np.mean(accs)

# Recall
def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> float:
    return recall_score(y_true=y_true, y_pred=y_pred, average=average, zero_division=0)

# Precision
def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> float:
    return precision_score(y_true=y_true, y_pred=y_pred, average=average, zero_division=0)

# F1-score
def f1_measure(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> float:
    return f1_score(y_true=y_true, y_pred=y_pred, average=average, zero_division=0)

# Hamming Loss
def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return hl(y_true=y_true, y_pred=y_pred)

# Subset Accuracy
def subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true=y_true, y_pred=y_pred)


# Ranking Loss
def ranking_loss(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    return label_ranking_loss(y_true=y_true, y_score=y_scores)


# One-error
def one_error(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    n_samples = y_true.shape[0]
    error = 0.0
    for i in range(n_samples):
        top_label = np.argmax(y_scores[i])
        if y_true[i, top_label] != 1:
            error += 1
    return error / n_samples


# Coverage
def coverage(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    return coverage_error(y_true=y_true, y_score=y_scores) - 1


# Average Precision
def average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    return average_precision_score(y_true=y_true, y_score=y_scores, average='weighted')
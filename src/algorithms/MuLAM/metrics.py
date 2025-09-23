import numpy as np

# Accuracy 
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    accuracy = []
    for label in range(y_true.shape[1]):
        accuracy.append(np.mean(y_true[:, label] == y_pred[:, label]))
    return np.mean(accuracy)

# Recall
def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recall = []
    for label in range(y_true.shape[1]):
        tp = np.sum((y_true[:, label] == 1) & (y_pred[:, label] == 1))
        fn = np.sum((y_true[:, label] == 1) & (y_pred[:, label] == 0))
        recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return np.mean(recall)

# Precision
def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    precision = []
    for label in range(y_true.shape[1]):
        tp = np.sum((y_true[:, label] == 1) & (y_pred[:, label] == 1))
        fp = np.sum((y_true[:, label] == 0) & (y_pred[:, label] == 1))
        precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return np.mean(precision)

# F1-score
def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> float:
    f1_scores = []
    for label in range(y_true.shape[1]):
        tp = np.sum((y_true[:, label] == 1) & (y_pred[:, label] == 1))
        fn = np.sum((y_true[:, label] == 1) & (y_pred[:, label] == 0))
        fp = np.sum((y_true[:, label] == 0) & (y_pred[:, label] == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * (precision * recall) / (precision + recall))
    
    if average == 'macro':
        return np.mean(f1_scores)
    elif average == 'micro':
        total_tp = np.sum((y_true == 1) & (y_pred == 1))
        total_fn = np.sum((y_true == 1) & (y_pred == 0))
        total_fp = np.sum((y_true == 0) & (y_pred == 1))
        
        total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        
        if total_precision + total_recall == 0:
            return 0.0
        else:
            return 2 * (total_precision * total_recall) / (total_precision + total_recall)
    else:
        raise ValueError(f'Unknown average type: {average}! average should be either "macro" or "micro".')
    

# Hamming Loss
def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true != y_pred)


# Subset Accuracy
def subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.all(y_true == y_pred, axis=1))


# Ranking Loss
def ranking_loss(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    n_samples, _ = y_true.shape
    loss = 0.0
    for i in range(n_samples):
        pos = np.where(y_true[i] == 1)[0]
        neg = np.where(y_true[i] == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        pairwise = 0
        for p in pos:
            for n in neg:
                if y_scores[i, p] <= y_scores[i, n]:
                    pairwise += 1
        loss += pairwise / (len(pos) * len(neg))
    return loss / n_samples


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
    n_samples, n_labels = y_true.shape
    cov = 0.0
    for i in range(n_samples):
        sorted_labels = np.argsort(-y_scores[i])
        true_labels = np.where(y_true[i] == 1)[0]
        if len(true_labels) == 0:
            continue
        max_pos = max(np.where(np.isin(sorted_labels, true_labels))[0])
        cov += max_pos
    return cov / n_samples / (n_labels - 1)


# Average Precision
def average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    n_samples = y_true.shape[0]
    avg_prec = 0.0
    for i in range(n_samples):
        sorted_labels = np.argsort(-y_scores[i])
        true_labels = np.where(y_true[i] == 1)[0]
        if len(true_labels) == 0:
            continue
        score = 0.0
        hits = 0
        for j, label in enumerate(sorted_labels, start=1):
            if label in true_labels:
                hits += 1
                score += hits / j
        avg_prec += score / len(true_labels)
    return avg_prec / n_samples
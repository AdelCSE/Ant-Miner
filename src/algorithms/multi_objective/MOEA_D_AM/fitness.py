import pandas as pd

def fitness_function(data: pd.DataFrame, rule: list) -> list:
    """
    Computes the fitness of a rule based on Sensitivity and Confidence.

    Args:
        data (pd.DataFrame): The dataset containing the attributes and class labels.
        rule (list): The rule to evaluate, represented as a list of tuples (attribute, value),
                     where the last tuple is assumed to be ('class', target_class).

    Returns:
        list: A list containing the fitness values [sensitivity, confidence], F1-score.
    """

    fitness_sensitivity = 0.0
    fitness_confidence = 0.0
    f1_score = 0.0

    if len(rule) == 0:
        return [fitness_sensitivity, fitness_confidence], f1_score
    

    target_class = rule[-1][1]
    conditions = rule[:-1]

    subset = data.copy()
    for term in conditions:
        subset = subset[subset[term[0]] == term[1]]

    # True Positives (TP): Rule matches and class is correct
    tp = len(subset[subset['class'] == target_class])

    # False Positives (FP): Rule matches but class is wrong
    fp = len(subset[subset['class'] != target_class])

    # False Negatives (FN): Class is correct but rule does not match
    fn = len(data[(data['class'] == target_class)]) - tp

    # Sensitivity (Recall) = TP / (TP + FN) 
    if (tp + fn) > 0:
        fitness_sensitivity = tp / (tp + fn)

    # Confidence (Precision) = TP / (TP + FP)
    if (tp + fp) > 0:
        fitness_confidence = tp / (tp + fp)

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    if (fitness_sensitivity + fitness_confidence) > 0:
        f1_score = 2 * (fitness_confidence * fitness_sensitivity) / (fitness_confidence + fitness_sensitivity)

    return [fitness_sensitivity, fitness_confidence], f1_score
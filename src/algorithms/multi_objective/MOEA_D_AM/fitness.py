import pandas as pd

def fitness_function(data: pd.DataFrame, ant: dict, labels: list[str], task: str) -> list:
    """
    Computes the fitness of a rule based on Sensitivity and Confidence.

    Args:
        data (pd.DataFrame): The dataset containing the attributes and class labels.
        rule (list): The rule to evaluate, represented as a list of tuples (attribute, value),
                     where the last tuple is assumed to be ('class', target_class).

    Returns:
        list: A list containing the fitness values [sensitivity, confidence], F1-score.
    """

    total_confidence, total_sensitivity, total_simplicity = [], [], []
    #total_specificity, total_f1_macro, total_f1_micro = [], [], []
    

    if task == 'single':
        rules = [ant['rule']] if len(ant['rule']) > 0 else []
    else:
        rules = [r['rule'] for r in ant['ruleset']['rules']] if len(ant['ruleset']['rules']) > 0 else []

    if len(rules) == 0:
        return [0.0, 0.0], 0.0
    
    for _, rule in enumerate(rules):
        if not rule:
            continue

        rule_confidence, rule_sensitivity= [], []
        #rule_specificity = []

        antecedent = [term for term in rule if term[0] not in labels]
        consequent = [term for term in rule if term[0] in labels]

        if len(consequent) == 0:
            continue

        subset = data.copy()
        for term in antecedent:
            subset = subset[subset[term[0]] == term[1]]

        for label, pred_val in consequent:

            # True Positives (TP): Rule matches and class is correct
            tp = len(subset[subset[label] == pred_val])

            # False Positives (FP): Rule matches but class is wrong
            fp = len(subset[subset[label] != pred_val])

            # False Negatives (FN): Class is correct but rule does not match
            fn = len(data[data[label] == pred_val]) - tp

            # Sensitivity (Recall) = TP / (TP + FN)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # Confidence (Precision) = TP / (TP + FP)
            confidence = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Specificity = TN / (TN + FP)
            #tn = len(subset[subset[label] != pred_val])
            #specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            rule_sensitivity.append(sensitivity)
            rule_confidence.append(confidence)
            #rule_specificity.append(specificity)

        
        avg_confidence = sum(rule_confidence) / len(rule_confidence) if rule_confidence else 0.0
        avg_sensitivity = sum(rule_sensitivity) / len(rule_sensitivity) if rule_sensitivity else 0.0
        #avg_specificity = sum(rule_specificity) / len(rule_specificity) if rule_specificity else 0.0
        f1_score = (2 * avg_sensitivity * avg_confidence) / (avg_sensitivity + avg_confidence) if (avg_sensitivity + avg_confidence) > 0 else 0.0



        simplicity = 1 / len(antecedent) if len(antecedent) > 0 else 0.0

        """
        if len(labels) > 1:
            ant['ruleset']['rules'][idx]['fitness'] = [simplicity, avg_confidence]
            ant['ruleset']['rules'][idx]['f1_score'] = f1_score
        """
        #avg_f1_macro = f1_macro(subset, data, consequent)
        #avg_f1_micro = f1_micro(subset, data, consequent)

        total_confidence.append(avg_confidence)
        total_sensitivity.append(avg_sensitivity)
        #total_specificity.append(avg_specificity)
        #total_f1_macro.append(avg_f1_macro)
        #total_f1_micro.append(avg_f1_micro)
        total_simplicity.append(simplicity)
    
    fitness_sensitivity = sum(total_sensitivity) / len(total_sensitivity) if total_sensitivity else 0.0
    fitness_confidence = sum(total_confidence) / len(total_confidence) if total_confidence else 0.0
    #fitness_specificity = sum(total_specificity) / len(total_specificity) if total_specificity else 0.0
    fitness_simplicity = sum(total_simplicity) / len(total_simplicity) if total_simplicity else 0.0
    #fitness_f1_macro = sum(total_f1_macro) / len(total_f1_macro) if total_f1_macro else 0.0
    #fitness_f1_micro = sum(total_f1_micro) / len(total_f1_micro) if total_f1_micro else 0.0

    f1_score = (2 * fitness_sensitivity * fitness_confidence) / (fitness_sensitivity + fitness_confidence) if (fitness_sensitivity + fitness_confidence) > 0 else 0.0

    return [fitness_confidence, fitness_simplicity], f1_score


def f1_macro(subset: pd.DataFrame, data: pd.DataFrame, consequent) -> float:
    total_f1 = []

    for label, pred_val in consequent:
        # True Positives (TP): Rule matches and class is correct
        tp = len(subset[subset[label] == pred_val])

        # False Positives (FP): Rule matches but class is wrong
        fp = len(subset[subset[label] != pred_val])

        # False Negatives (FN): Class is correct but rule does not match
        fn = len(data[data[label] == pred_val]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        total_f1.append(f1)

    macro_f1 = sum(total_f1) / len(total_f1) if total_f1 else 0.0
    return macro_f1


def f1_micro(subset: pd.DataFrame, data: pd.DataFrame, consequent) -> float:
    tp, fp, fn = 0, 0, 0

    for label, pred_val in consequent:
        # True Positives (TP): Rule matches and class is correct
        tp += len(subset[subset[label] == pred_val])

        # False Positives (FP): Rule matches but class is wrong
        fp += len(subset[subset[label] != pred_val])

        # False Negatives (FN): Class is correct but rule does not match
        fn += len(data[data[label] == pred_val]) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    micro_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return micro_f1

def hamming_loss_ml(subset: pd.DataFrame, data: pd.DataFrame, consequent) -> float:
    total_labels = len(consequent)
    if total_labels == 0:
        return 1.0  # Maximum loss if no labels are predicted

    incorrect_predictions = 0

    for label, pred_val in consequent:
        # Count mismatches between predicted and actual values
        incorrect_predictions += len(data[data[label] != pred_val])

    hamming_loss = incorrect_predictions / (len(data) * total_labels)
    return hamming_loss
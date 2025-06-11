import pandas as pd
import numpy as np
import math

def get_class_probs(data : pd.DataFrame, attr_name : str, attr_value : str, class_label : str) -> list:
    """
    Calculate the class probabilities for a given attribute value.
    """

    subset = data[data[attr_name] == attr_value]
    total = len(subset)
    if total == 0:
        return []
        
    class_counts = subset[class_label].value_counts().tolist()
    probs = [count / total for count in class_counts]
    return probs


def compute_entropy(probs : list) -> float:
    """
    Compute the entropy of a list of probabilities.
    """
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy


def check_attributes_left(attrs : list, sub_attrs : list) -> bool:
    """
    Check if there are any attributes left that are not used in a rule.
    """
    for attr in attrs:
        if attr not in sub_attrs:
            return True
        
    return False


def assign_class(data : pd.DataFrame, rule : list) -> list:
    """
    Assign a class to a rule based on the data subset that matches the rule.
    """
    subset = data.copy()
    for term in rule:
        subset = subset[subset[term[0]] == term[1]]
    
    assigned_class = subset['class'].value_counts().idxmax()
    rule += [('class', assigned_class)]

    return rule
    

def roulette_wheel(probabilities):
    cumulative_sum = np.cumsum(probabilities)
    r = np.random.rand()
    for i, cs in enumerate(cumulative_sum):
        if r < cs:
            return i
    return len(probabilities) - 1  # Fallback in case of floating-point errors

def rule_covers_min_examples(data : pd.DataFrame, rule : list, threshold : int) -> bool:
    """
    Check if a rule covers at least a minimum number of examples in the dataset.
    """
    subset = data.copy()
    for term in rule:
        subset = subset[subset[term[0]] == term[1]]

    return len(subset) >= threshold


def drop_covered(best_rule : list, data : pd.DataFrame) -> pd.DataFrame: 
    """
    Drop the instances covered by the best rule.
    """
    subset = data.copy()
    for term in best_rule:
        subset = subset[subset[term[0]] == term[1]]

    data = data.drop(subset.index).reset_index(drop=True)

    return data
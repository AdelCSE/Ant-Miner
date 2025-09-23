import numpy as np
import pandas as pd
import math
from scipy.stats import chi2

def get_label_probs(subset : pd.DataFrame, attr : str, value : str, label : str) -> list:
    """
    Calculate the class probabilities for a given attribute value.
    """
    subset = subset[subset[attr] == value]
    total = len(subset)
    if total == 0:
        return []
    
    class_counts = subset[label].value_counts().tolist()
    probs = [count / total for count in class_counts]
    return probs

def compute_entropy(probs):
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy


def calculate_terms_probs(terms : list, 
                          labels : list[str],
                          pheromones : dict, 
                          heuristics : dict, 
                          unused_attrs : list, 
                          alpha : int, 
                          beta : int
                          ) -> list:
    probs = []
    for term in terms:
        if term[0] in unused_attrs:
            # sum pheromone values across all labels (consider other strategies?)
            tau = sum(pheromones[label][term] for label in labels) ** alpha
            eta = heuristics[term] ** beta
            probs.append(tau * eta)
    probs_sum = sum(probs)
    probs = [p / probs_sum for p in probs] if probs_sum > 0 else probs

    return probs


def select_term(terms: list, attrs: list, p : list) -> tuple:
    
    available_terms = [term for term in terms if term[0] in attrs]

    indices = np.arange(len(available_terms))
    selected_index = np.random.choice(indices, p=p)
    selected_term = available_terms[selected_index]

    return selected_term


def covers_min_examples(data : pd.DataFrame, rule : list, threshold : int) -> bool:
    subset = data.copy()
    for term in rule:
        subset = subset[subset[term[0]] == term[1]]

    return len(subset) >= threshold


def build_contingency(subset: pd.DataFrame, covered_examples: pd.DataFrame, label: str):
    """
    Build contingency table for antecedent coverage vs class values.
    Returns a 2D numpy array: rows=[covered, not covered], cols=class values.
    """
    values = subset[label].unique()
    table = []
    
    for v in values:
        count_covered = np.sum(covered_examples[label] == v)
        count_not_covered = np.sum((~subset.index.isin(covered_examples.index)) &
                                   (subset[label] == v))
        table.append([count_covered, count_not_covered])
    
    return np.array(table).T


def chi2_stat(contingency):
    """
    Compute chi-square statistic from contingency table.
    """
    observed = contingency
    row_sums = observed.sum(axis=1)[:, None]
    col_sums = observed.sum(axis=0)[None, :]
    total = observed.sum()
    expected = row_sums @ col_sums / total
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2_val = np.nansum((observed - expected) ** 2 / expected)
    return chi2_val


def compute_cramers_v(contingency):
    """
    Compute Cramer's V from contingency table.
    """
    chi2_val = chi2_stat(contingency)
    n = np.sum(contingency)
    k = min(contingency.shape)  # min(rows, cols)
    return np.sqrt(chi2_val / (n * (k - 1))) if n > 0 and k > 1 else 0.0


def compute_v_threshold(subset: pd.DataFrame, covered_examples: pd.DataFrame, selected_terms: tuple, label: str, contingency) -> float:
    parent_examples = len(subset) 
    num_children = np.prod([len(subset[term[0]].unique()) for term in selected_terms])
    expected_size = parent_examples / num_children
    
    chi2_critical = chi2.ppf(1 - 0.05, 1)
    k = min(len(covered_examples[label].unique()), contingency.shape[1])
    V_threshold = np.sqrt(chi2_critical / (expected_size * (k - 1))) if (k > 1 and expected_size > 0) else 1.0

    return V_threshold


def assign_classes(data : pd.DataFrame, rule_antecedent : list, labels: list[str]) -> list:
    subset = data.copy()
    rule_consequent = []
    for term in rule_antecedent:
        subset = subset[subset[term[0]] == term[1]]
    
    for label in labels:
        assigned_class = subset[label].value_counts().idxmax()
        rule_consequent.append((label, assigned_class))

    return rule_consequent


def assign_class_for_label(data : pd.DataFrame, rule_antecedent : list, label: str) -> list:
    subset = data.copy()
    for term in rule_antecedent:
        subset = subset[subset[term[0]] == term[1]]
    
    assigned_class = subset[label].value_counts().idxmax()
    rule_consequent = (label, assigned_class)

    return rule_consequent


def evaluate_rule(rule : list, data : pd.DataFrame, label : str, labels: list[str]) -> float:
    subset = data.copy()

    if len(rule) == 0:
        return 0.0, [0.0, 0.0]

    for term in rule:
        if term[0] not in labels:
            subset = subset[subset[term[0]] == term[1]]

    tp = len(subset[subset[label] == rule[-1][1]])
    fp = len(subset[subset[label] != rule[-1][1]])
    fn = len(data[data[label] == rule[-1][1]]) - tp
    tn = len(data) - tp - fp - fn
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    quality = sensitivity * specificity

    # confidence = Laplace precision
    confidence = (tp + 1) / (tp + fp + 2)

    return quality, confidence
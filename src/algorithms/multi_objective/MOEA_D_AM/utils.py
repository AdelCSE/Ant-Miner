import pandas as pd
import numpy as np
import math
from scipy.stats import chi2

def get_gmax(colony : dict, weights: np.ndarray, reference: np.ndarray, task: str, approach : str ="weighted"):

    g_values = []

    for i, ant in enumerate(colony['ants']):
        if task == 'single':
            fitness = ant['fitness']
        else:
            fitness = ant['ruleset']["fitness"]

        if approach == "weighted":
            g = sum(weights[i][l] * fitness[l] for l in range(len(fitness)))
        elif approach == "tchebycheff":
            if reference is None:
                raise ValueError("Reference point (z*) must be provided for Tchebycheff approach.")
            g = max(weights[i][j] * abs(fitness[j] - reference[j]) for j in range(len(fitness)))
        else:
            raise ValueError(f"Unknown decomposition approach: {approach}! Use 'weighted' or 'tchebycheff'.")
        
        g_values.append(g)
   
    return max(g_values) if g_values else 0.0


def update_reference_point(reference_point, colony, task: str, maximize=True):
    """
    Update reference_point (z*) based on the new colony solutions.
    """
    m = len(reference_point)
    for ant in colony['ants']:
        for j in range(m):
            if task == 'single':
                fitness = ant['fitness']
            else:
                fitness = ant['ruleset']['fitness']

            if maximize:
                reference_point[j] = max(reference_point[j], fitness[j])
            else:
                reference_point[j] = min(reference_point[j], fitness[j])
    return reference_point


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


def assign_class(data : pd.DataFrame, rule_antecedent : list, label: str) -> list:
    """
    Assign a class to a rule based on the data subset that matches the rule.
    """
    subset = data.copy()
    for term in rule_antecedent:
        subset = subset[subset[term[0]] == term[1]]
    
    assigned_class = subset[label].value_counts().idxmax()

    return [(label, assigned_class)]
    

def rule_covers_min_examples(data : pd.DataFrame, rule : list, threshold : int) -> bool:
    """
    Check if a rule covers at least a minimum number of examples in the dataset.
    """
    subset = data.copy()
    for term in rule:
        subset = subset[subset[term[0]] == term[1]]

    return len(subset) >= threshold


def drop_covered(best_ant : dict, data : pd.DataFrame, task: str) -> pd.DataFrame: 
    """
    Drop the instances covered by the best ant.
    """
    

    if task == 'single':
        rules = [best_ant['rule']]
    else:
        rules = [rule['rule'] for rule in best_ant['ruleset']['rules'] if len(rule['rule']) > 0]
        
    for best_rule in rules:
        subset = data.copy()
        for term in best_rule:
            subset = subset[subset[term[0]] == term[1]]
            
        data = data.drop(subset.index).reset_index(drop=True)

    return data


def remove_dominated_rules(archive):
    """
    Remove dominated rules from the archive.
    """
    non_dominated = []
    for rule in archive:
        if not any(
            all(r1 >= r2 for r1, r2 in zip(rule['fitness'], other_rule['fitness'])) and
            any(r1 > r2 for r1, r2 in zip(rule['fitness'], other_rule['fitness']))
            for other_rule in archive if other_rule != rule
        ):
            non_dominated.append(rule)

    return non_dominated


def get_term_rule_ratio(archive: list, archive_type: str, labels: list[str], task: str) -> float:
    """
    Calculate the ratio of terms per rule in the archive.
    """

    if task == 'single':

        if archive_type == 'rulesets':
            terms = sum(len(item['rule']) - len(labels) for solution in archive for item in solution['ruleset'])
            rules = sum(len(solution['ruleset']) for solution in archive)
        elif archive_type == 'rules':
            terms = sum(len(solution['rule']) - len(labels) for solution in archive)
            rules = len(archive)
        else:
            raise ValueError("Invalid archive type. Expected 'rulesets' or 'rules'.")
        
    else:
        rule_terms = 0
        rule_count = 0
        for solution in archive:
            for rule in solution['ruleset']['rules']:
                rule_labels = [term for term in rule['rule'] if term[0] in labels]
                rule_terms += len(rule['rule']) - len(rule_labels)
                rule_count += 1
        terms = rule_terms
        rules = rule_count
        if rules == 0:
            return 0.0

    return terms / rules if rules > 0 else 0.0




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
    expected_size = parent_examples 
    
    df = (contingency.shape[0] - 1) * (contingency.shape[1] - 1)
    chi2_critical = chi2.ppf(1 - 0.05, df)
    k = min(len(covered_examples[label].unique()), contingency.shape[1])
    V_threshold = np.sqrt(chi2_critical / (expected_size * (k - 1))) if (k > 1 and expected_size > 0) else 1.0

    return V_threshold

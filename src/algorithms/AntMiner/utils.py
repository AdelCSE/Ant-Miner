import pandas as pd
import numpy as np
import math

def rule_covers_min_examples(data : pd.DataFrame, 
                             rule : list, 
                             threshold : int
                             ) -> bool:
    subset = data.copy()
    for term in rule:
        subset = subset[subset[term[0]] == term[1]]

    return len(subset) >= threshold


def check_attributes_left(attrs : list, 
                          sub_attrs : list
                          ) -> bool:
    for attr in attrs:
        if attr not in sub_attrs:
            return True
        
    return False


def calculate_terms_probs(terms : list, 
                          pheromones : dict, 
                          heuristics : dict, 
                          used_attrs : list, 
                          alpha : int, 
                          beta : int
                          ) -> list:
    probs = []
    for term in terms:
        tau = pheromones[term] ** alpha
        eta = heuristics[term] ** beta
        x = 0 if term[0] in used_attrs else 1
        probs.append(x * tau * eta)
    probs_sum = sum(probs)
    probs = [p / probs_sum for p in probs]

    return probs


def compute_entropy(probs):
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy


def select_term(terms : list, 
                used_attrs: list, 
                p : list
                ) -> tuple:
    indices = np.arange(len(terms))
    selected_index = np.random.choice(indices, p=p)

    # TODO: handle the case where it get stuck in the loop
    while terms[selected_index][0] in used_attrs:
        selected_index = np.random.choice(terms, p=p)
    selected_term = terms[selected_index]

    return selected_term
    

def assign_class(data : pd.DataFrame, 
                 rule : list
                 ) -> list:
    subset = data.copy()
    for term in rule:
        subset = subset[subset[term[0]] == term[1]]
    
    # TODO: remove bool when classes are string
    assigned_class = bool(subset['class'].value_counts().idxmax())
    rule += [('class', assigned_class)]

    return rule


def evaluate_rule(rule : list, 
                  data : pd.DataFrame
                  ) -> float:
    subset = data.copy()

    for term in rule[:-1]:
        subset = subset[subset[term[0]] == term[1]]

    tp = len(subset[subset['class'] == rule[-1][1]])
    fp = len(subset[subset['class'] != rule[-1][1]])
    fn = len(data[data['class'] == rule[-1][1]]) - tp
    tn = len(data) - tp - fp - fn
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return sensitivity * specificity
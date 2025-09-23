import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
# import sensitivity and specificity metrics
from sklearn.metrics import confusion_matrix


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
        if term[0] not in used_attrs:
            tau = pheromones[term] ** alpha
            eta = heuristics[term] ** beta
            probs.append(tau * eta)
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
    

    # remove the terms that are already used
    available_terms = [term for term in terms if term[0] not in used_attrs]

    indices = np.arange(len(available_terms))
    selected_index = np.random.choice(indices, p=p)
    selected_term = available_terms[selected_index]

    return selected_term
    

def assign_class(data : pd.DataFrame, 
                 rule : list
                 ) -> list:
    subset = data.copy()
    for term in rule:
        subset = subset[subset[term[0]] == term[1]]
    
    # TODO: remove bool when classes are string
    assigned_class = subset['class'].value_counts().idxmax()
    rule += [('class', assigned_class)]

    return rule


def evaluate_rule(rule : list, data : pd.DataFrame, label: str) -> float:
    subset = data.copy()

    if len(rule) == 0:
        return 0.0, [0.0, 0.0]

    for term in rule[:-1]:
        subset = subset[subset[term[0]] == term[1]]

    tp = len(subset[subset[label] == rule[-1][1]])
    fp = len(subset[subset[label] != rule[-1][1]])
    fn = len(data[data[label] == rule[-1][1]]) - tp
    tn = len(data) - tp - fp - fn
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return sensitivity * specificity, [sensitivity, specificity]

def plot_patero_front(archive: list) -> None:

    if len(archive) == 0:
        raise ValueError("Archive is empty. No solutions to plot!")

    fitness_values = np.array(archive)
    print(fitness_values)

    plt.figure(figsize=(8, 6))
    plt.scatter(fitness_values[:, 0], fitness_values[:, 1], c='blue', marker='o', label='Non-dominated solutions')
    plt.title('Pareto Front of Non-dominated Solutions')
    plt.xlabel('sensitivity')
    plt.ylabel('specificity')
    plt.grid(True)
    plt.legend()
    plt.show()


def dominates(fitness1, fitness2):
    """Returns True if fitness1 dominates fitness2."""
    return all(f1 >= f2 for f1, f2 in zip(fitness1, fitness2)) and any(f1 > f2 for f1, f2 in zip(fitness1, fitness2))

def update_EP(rules, fitnesses, EP):
    """
    Updates the external Pareto archive (EP) with non-dominated solutions from the colony.

    Args:
        colony (dict): Should contain a list of ants under `colony["ant"]`, each with a `fitness` list.
        EP (list): Existing archive of non-dominated solutions (list of ant dicts).

    Returns:
        ant_best_rule (list[int]): Indices of ants that produced new non-dominated solutions.
        EP (list): Updated Pareto archive.
    """
    for i, ant in enumerate(rules):
        dominated = False
        for ep_ant in EP:
            if dominates(ep_ant["fitness"], fitnesses[i]):
                dominated = True
                break

        if not dominated:
            # Remove all EP ants dominated by current ant
            EP = [ep_ant for ep_ant in EP if not dominates(fitnesses[i], ep_ant["fitness"])]

            # Add the new non-dominated solution

            if not any(set(ep_ant["rule"]) == set(ant) for ep_ant in EP):
                EP.append({'rule': ant, 'fitness': fitnesses[i]})

    return EP
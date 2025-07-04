import numpy as np
import pandas as pd
from .utils import check_attributes_left, assign_class, rule_covers_min_examples


def create_colony(data : pd.DataFrame, attributes : list, terms : list, population : int, gamma : float, phi :  dict, min_examples: int, random_state: int = None) -> dict:
    """
    Create a colony of ants with rules based on the given attributes and terms.
    
    Args:
        data (pd.DataFrame): The dataset to be used for rule generation.
        attributes (list): List of attributes to be used in rules.
        terms (list): List of terms available for rules, each term is a tuple (attribute, value).
        population (int): Number of ants in the colony.
        gamma (float): Probability threshold to use greedy selection.
        phi (dict): desirability matrix for terms, shape (population, terms).
        min_examples (int): Minimum number of examples a rule must cover.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        dict: Colony containing ants with their rules.
    """
    
    np.random.seed(random_state)
    colony = {'ants': []}

    for i in range(population):

        rule = []
        used_attrs = []

        while check_attributes_left(attrs=attributes, sub_attrs=used_attrs):

            available_terms = [term for term in terms if term[0] not in used_attrs]
            
            if np.random.rand() > gamma:
                # Greedy selection
                phi_values = np.array([phi[i][term] for term in available_terms])
                max_index = np.argmax(phi_values)
                next_term = available_terms[max_index]
            
            else:

                # Probabilistic selection (roulette wheel)
                probs = np.array([phi[i][term] for term in available_terms])
                prob_sum = np.sum(probs)
                indices = np.arange(len(available_terms))
    
                probs /= prob_sum
                next_index = np.random.choice(indices, p=probs)
                next_term = available_terms[next_index]

            if not rule_covers_min_examples(data=data, rule=rule + [next_term], threshold=min_examples):
                break

            used_attrs.append(next_term[0])
            rule.append(next_term)

        if len(rule) > 0:
            rule = assign_class(data=data, rule=rule)

        colony['ants'].append({'rule': rule})

    return colony
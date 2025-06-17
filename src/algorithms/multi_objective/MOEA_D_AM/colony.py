import numpy as np
import pandas as pd
from .utils import check_attributes_left, roulette_wheel, assign_class, rule_covers_min_examples

import pprint

def create_colony(data : pd.DataFrame, attributes : list, terms : list, population : int, gamma : float, phi :  dict, min_examples: int) -> dict:
    """
    Create a colony of ants with probabilistic or greedy construction of tours.
    
    Args:
        attributes (list): List of attributes to be used in rules.
        terms (list): List of terms available for rules, each term is a tuple (attribute, value).
        population (int): Number of ants in the colony.
        gamma (float): Probability threshold to use greedy selection.
        phi (dict): desirability matrix for terms, shape (population, terms).

    Returns:
        dict: colony dictionary with key 'ant' and a list of ants, each with a 'rule'.
    """
    
    colony = {'ants': []}

    for i in range(population):

        rule = []
        used_attrs = []

        rejection_count = 0

        indices = np.arange(len(terms))
        selected_index = np.random.choice(indices)
        initial_term = terms[selected_index]

        # to ensure the rule contains at least one term
        while not rule_covers_min_examples(data=data, rule=[initial_term], threshold=min_examples):
            selected_index = np.random.choice(indices)
            initial_term = terms[selected_index]

            # to avoid infinite loop in case of no valid initial term
            rejection_count += 1
            if rejection_count > 100:
                break

        if rejection_count > 100:
            colony['ants'].append({'rule': []})
            continue

        rule.append(initial_term)
        used_attrs.append(initial_term[0])  

        while check_attributes_left(attrs=attributes, sub_attrs=used_attrs):

            if not rule_covers_min_examples(data=data, rule=rule, threshold=min_examples):
                used_attrs.pop()
                rule.pop()
                break

            available_terms = [term for term in terms if term[0] not in used_attrs]
            """
            if np.random.rand() < gamma:
                # Greedy selection
                phi_values = np.array([phi[i][term] for term in available_terms])
                max_index = np.argmax(phi_values)
                next_term = available_terms[max_index]
            
            else:
            """

            # Probabilistic selection (roulette wheel)
            probs = np.array([phi[i][term] for term in available_terms])
            prob_sum = np.sum(probs)
            indices = np.arange(len(available_terms))

            probs /= prob_sum
            next_index = np.random.choice(indices, p=probs)
            next_term = available_terms[next_index]

            used_attrs.append(next_term[0])
            rule.append(next_term)

        rule = assign_class(data=data, rule=rule)
        colony['ants'].append({'rule': rule})

    return colony
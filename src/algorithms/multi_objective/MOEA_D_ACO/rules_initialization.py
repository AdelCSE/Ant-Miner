import numpy as np
import pandas as pd
from .utils import check_attributes_left, assign_class, rule_covers_min_examples

def initialize_rules(colony: dict, data: pd.DataFrame, attributes: list, terms: list, population: int, min_examples: int) -> dict:
    """
    Initializes a colony of ants with random rules.

    Args:
        colony (dict): Dictionary containing colony state, will be updated.
        attributes (list): List of attributes to be used in rules.
        terms (list): List of terms available for rules, each term is a tuple (attribute, value).
        population (int): Number of ants in the colony.

    Returns:
        dict: Updated colony with initialized ants and their rules.
    """

    colony = {'ant': []}

    for i in range(population):

        used_attrs = []
        rule = []

        while check_attributes_left(attrs=attributes, sub_attrs=used_attrs):
            available_terms = [term for term in terms if term[0] not in used_attrs]
            indices = np.arange(len(available_terms))
            selected_index = np.random.choice(indices)
            selected_term = available_terms[selected_index]

            if not rule_covers_min_examples(data=data, rule=rule + [selected_term], threshold=min_examples):
                break

            used_attrs.append(selected_term[0])
            rule.append(selected_term)

        # Assign class to the rule
        rule = assign_class(data=data, rule=rule)

        colony['ant'].append({'rule': rule})


    return colony
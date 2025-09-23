import numpy as np
import pandas as pd
from .utils import assign_class, rule_covers_min_examples
from .utils import build_contingency, chi2_stat, compute_cramers_v, compute_v_threshold


def create_colony(task : str, data : pd.DataFrame, attributes : list[str], labels : list[str], terms : list, population : int, gamma : float, phi :  dict, min_examples: int, random_state: int = None) -> dict:
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

        # for multi-label task
        ruleset = {'rules': []}

        unused_attrs = attributes.copy()
        unpredicted_labels = labels.copy()

        rule_antecedent, rule_consequent = [], []

        while len(unused_attrs) > 0 and (task == 'single' or (task == 'multi' and len(unpredicted_labels) > 0)):

            available_terms = [term for term in terms if term[0] in unused_attrs]
            
            if np.random.rand() > gamma:
                # Greedy selection
                phi_values = np.array([phi[i][term] for term in available_terms])
                next_term = available_terms[np.argmax(phi_values)]
            else:
                # Probabilistic selection (roulette wheel)
                probs = np.array([phi[i][term] for term in available_terms])
                probs /= np.sum(probs) if np.sum(probs) > 0 else 1
                
                indices = np.arange(len(available_terms))
                next_index = np.random.choice(indices, p=probs)
                next_term = available_terms[next_index]

            if not rule_covers_min_examples(data=data, rule=rule_antecedent + [next_term], threshold=min_examples):
                break

            rule_antecedent.append(next_term)
            unused_attrs.remove(next_term[0])

        labels_to_remove = []

        if task == 'single':
            if len(rule_antecedent) > 0:
                rule_consequent = assign_class(data=data, rule_antecedent=rule_antecedent, label=labels[0])
                complete_rule = rule_antecedent + rule_consequent
                colony['ants'].append({'rule': complete_rule})
            else:
                colony['ants'].append({'rule': []})

        else:
            if len(rule_antecedent) > 0:
                covered_examples = data.copy()
                for term in rule_antecedent:
                    covered_examples = covered_examples[covered_examples[term[0]] == term[1]]

                scores = []
                
                for label in unpredicted_labels:
                    if len(covered_examples) == 0:
                        break

                    #contingency = build_contingency(data, covered_examples, label)
                    #V = compute_cramers_v(contingency)
                    #V_threshold = compute_v_threshold(data, covered_examples, rule_antecedent, label, contingency)
                    confidence = len(covered_examples[covered_examples[label] == '1']) / len(covered_examples) if len(covered_examples) > 0 else 0.0
                    interval = abs(confidence - 0.5) * 2
                    if interval >= 0.5:
                        val = '1' if confidence >= 0.75 else '0'
                        rule_consequent.append((label, val))
                        scores.append((label, confidence))
                        labels_to_remove.append(label)

                # Remove labels that have been predicted
                unpredicted_labels = [label for label in unpredicted_labels if label not in labels_to_remove]

                complete_rule = rule_antecedent + rule_consequent
                ruleset['rules'].append({'rule': complete_rule , 'scores': scores})
                
        if task == 'multi' and len(unpredicted_labels) > 0:
            for label in unpredicted_labels:
                unused_attrs = attributes.copy()
                rule_antecedent, rule_consequent = [], []

                while len(unused_attrs) > 0:
                    
                    available_terms = [term for term in terms if term[0] in unused_attrs]
                    
                    if np.random.rand() > gamma:
                        # Greedy selection
                        phi_values = np.array([phi[i][term] for term in available_terms])
                        next_term = available_terms[np.argmax(phi_values)]
                    else:
                        # Probabilistic selection (roulette wheel)
                        probs = np.array([phi[i][term] for term in available_terms])
                        probs /= np.sum(probs) if np.sum(probs) > 0 else 1
                        
                        indices = np.arange(len(available_terms))
                        next_index = np.random.choice(indices, p=probs)
                        next_term = available_terms[next_index]

                    if not rule_covers_min_examples(data=data, rule=rule_antecedent + [next_term], threshold=min_examples):
                        break

                    rule_antecedent.append(next_term)
                    unused_attrs.remove(next_term[0])

                if len(rule_antecedent) > 0:
                    rule_consequent = assign_class(data=data, rule_antecedent=rule_antecedent, label=label)
                    complete_rule = rule_antecedent + rule_consequent

                    # calculate confidence score
                    covered_examples = data.copy() 
                    for term in rule_antecedent:
                        covered_examples = covered_examples[covered_examples[term[0]] == term[1]]
                    confidence = len(covered_examples[covered_examples[label] == rule_consequent[0][1]]) / len(covered_examples) if len(covered_examples) > 0 else 0.0
                    ruleset['rules'].append({'rule': complete_rule , 'scores': [(label, confidence)]})


        if task == 'multi':
            colony['ants'].append({'ruleset': ruleset})
        
    return colony


import numpy as np
import pandas as pd
from .utils import assign_class, rule_covers_min_examples

def create_colony(data: pd.DataFrame, 
                  attributes: list[str], 
                  labels: list[str], 
                  terms: list, 
                  population: int, 
                  gamma: float, 
                  phi: dict, 
                  min_examples: int, 
                  ants_per_subproblem: int = 1,
                  random_state: int = None) -> dict:
    """
    Create a colony of ants for Single-Label Classification.
    For each subproblem (population index), it generates 'ants_per_subproblem' candidate rules.
    
    Args:
        data (pd.DataFrame): Dataset.
        attributes (list): Attributes to use.
        labels (list): Target label (list of 1 element for single-label).
        terms (list): Available (attr, val) terms.
        population (int): Number of subproblems/groups.
        gamma (float): Greedy selection probability.
        phi (dict): Desirability matrix.
        min_examples (int): Coverage constraint.
        ants_per_subproblem (int): Number of candidates to generate per subproblem.
        random_state (int): Seed.

    Returns:
        dict: Colony where colony['ants'][i] is a LIST of candidate ants for subproblem i.
    """
    
    np.random.seed(random_state)
    colony = {'ants': []}

    # 1. Iterate over each subproblem (Decomposition Weight Vector)
    for i in range(population):
        
        subproblem_candidates = []

        # 2. Inner Loop: Generate multiple candidates for THIS subproblem
        for _ in range(ants_per_subproblem):

            unused_attrs = attributes.copy()
            rule_antecedent = []
            
            # --- Ant Construction Phase ---
            while len(unused_attrs) > 0:
                available_terms = [term for term in terms if term[0] in unused_attrs]
                
                # Selection using phi[i] (specific to this subproblem)
                if np.random.rand() > gamma:
                    # Greedy
                    phi_values = np.array([phi[i][term] for term in available_terms])
                    next_term = available_terms[np.argmax(phi_values)]
                else:
                    # Probabilistic (Roulette Wheel)
                    probs = np.array([phi[i][term] for term in available_terms])
                    probs /= np.sum(probs) if np.sum(probs) > 0 else 1
                    
                    indices = np.arange(len(available_terms))
                    next_term = available_terms[np.random.choice(indices, p=probs)]

                # Check Coverage Constraint (Pruning during construction)
                if not rule_covers_min_examples(data=data, rule=rule_antecedent + [next_term], threshold=min_examples):
                    break

                rule_antecedent.append(next_term)
                unused_attrs.remove(next_term[0])

            # --- Consequent Assignment ---
            if len(rule_antecedent) > 0:
                # Assign class based on majority class of covered examples
                rule_consequent = assign_class(data=data, rule_antecedent=rule_antecedent, label=labels[0])
                complete_rule = rule_antecedent + rule_consequent
                candidate_ant = {'rule': complete_rule}
            else:
                # Empty rule case
                candidate_ant = {'rule': []}

            # Add this candidate to the pool
            subproblem_candidates.append(candidate_ant)

        # 3. Store the list of candidates for subproblem i
        colony['ants'].append(subproblem_candidates)

    return colony
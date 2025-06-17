import pandas as pd
from .fitness import fitness_function
from .utils import assign_class

def prune_rule(data : pd.DataFrame, rule : list) -> list:
    
    """
    Prune the rule by removing terms until no further improvement is possible.
    """
    
    best_quality = fitness_function(data=data, rule=rule)
    pruned_rule = rule[:-1]
    
    while len(pruned_rule) > 1:

        best_term_to_remove = None
        best_improvement = best_quality

        for i, term in enumerate(pruned_rule):
            temp_rule = pruned_rule[:i] + pruned_rule[i+1:]
            temp_rule = assign_class(data=data, rule=temp_rule)
            quality = fitness_function(data=data, rule=temp_rule)

            # dominance check
            if all(f1 >= f2 for f1, f2 in zip(quality, best_quality)) and any(f1 > f2 for f1, f2 in zip(quality, best_quality)):
                best_improvement = quality
                best_term_to_remove = term

        if best_term_to_remove:
            pruned_rule.remove(best_term_to_remove)
            best_quality = best_improvement
        else:
            break

    pruned_rule = assign_class(data=data, rule=pruned_rule)

    return pruned_rule
import pandas as pd
from .fitness import fitness_function
from .utils import assign_class

def prune_rule(data : pd.DataFrame, ant : dict, task: str, labels: list[str]) -> list:

    """
    Prune the rule by removing terms until no further improvement is possible.
    """

    if task == 'single':
        # ant is a dict with 'rule' and 'fitness'
        best_quality, _ = fitness_function(data=data, ant=ant, labels=['class'], task=task)
        pruned_rule = ant['rule'][:-1]
    
        while len(pruned_rule) > 1:
    
            best_term_to_remove = None
            best_improvement = best_quality
    
            for i, term in enumerate(pruned_rule):
                temp_rule = pruned_rule[:i] + pruned_rule[i+1:]
                class_label = assign_class(data=data, rule_antecedent=temp_rule, label='class')
                temp_rule = temp_rule + class_label
                quality, _ = fitness_function(data=data, ant={'rule': temp_rule}, labels=['class'], task=task)
    
                # dominance check
                if all(f1 >= f2 for f1, f2 in zip(quality, best_quality)) and any(f1 > f2 for f1, f2 in zip(quality, best_quality)):
                    best_improvement = quality
                    best_term_to_remove = term
    
            if best_term_to_remove:
                pruned_rule.remove(best_term_to_remove)
                best_quality = best_improvement
            else:
                break
    
        assigned_class = assign_class(data=data, rule_antecedent=pruned_rule, label='class')
        pruned_rule = pruned_rule + assigned_class

        return pruned_rule

    else:
        # ant is a dict with 'ruleset' and 'fitness'
        pruned_ruleset = {'rules': []}
        for j, rule in enumerate(ant['ruleset']['rules']):
            best_quality, _ = fitness_function(data=data, ant=ant, labels=labels, task=task)
            pruned_rule = [t for t in rule if t[0] not in labels]
            consequent = [t for t in rule if t[0] in labels]
            while len(pruned_rule) > 0:
    
                best_term_to_remove = None
                best_improvement = best_quality
    
                for i, term in enumerate(pruned_rule):
                    temp_rule = pruned_rule[:i] + pruned_rule[i+1:] + consequent
                    quality, _ = fitness_function(data=data, ant=ant, labels=labels, task=task)
    
                    # dominance check
                    if all(f1 >= f2 for f1, f2 in zip(quality, best_quality)) and any(f1 > f2 for f1, f2 in zip(quality, best_quality)):
                        best_improvement = quality
                        best_term_to_remove = term
    
                if best_term_to_remove:
                    pruned_rule.remove(best_term_to_remove)
                    best_quality = best_improvement
                else:
                    break
            pruned_rule = pruned_rule + consequent
            pruned_ruleset['rules'].append({'rule': pruned_rule, 'scores': ant['ruleset']['rules'][j]['scores']})

        return pruned_ruleset

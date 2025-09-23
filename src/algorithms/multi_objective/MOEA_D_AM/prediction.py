import pandas as pd
import numpy as np

def matches_rule(row, rule):
    """
    Checks if a given row matches the a given rule.
    """
    return all(row[term[0]] == term[1] for term in rule[:-1])


def predict(X, rules):
    """
    Predicts the class labels for the given data using the provided rules.
    """
    preds = []
    triggered = {
        'rules': [],
        'rule_ids': [],
        'confidences': []
    }
    for idx, row in X.iterrows():
        matched = False
        for ant in rules:
            if matches_rule(row, ant['rule']):
                triggered['rules'].append(ant['rule'])
                triggered['rule_ids'].append(f'rule{idx}')
                triggered['confidences'].append(ant['fitness'][0])
                matched = True
                break

        preds.append('pos' if matched else 'neg')
        if not matched:
            triggered['rules'].append('No Match - Default to neg')
            triggered['rule_ids'].append(f'rule{idx}')
            triggered['confidences'].append(1.0)

    return preds, triggered


def predict_mlc(archive, data: pd.DataFrame, labels: list[str], priors: dict[str, float]) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Predict the classes for new instances based on the discovered rulesets.
    """
    n_samples = len(data)
    n_labels = len(labels)
    predictions = np.zeros((n_samples, n_labels), dtype=int)
    scores = np.zeros((n_samples, n_labels), dtype=float)
    # sort rulesets by quality
    archive = sorted(archive, key=lambda x: x['ruleset']['f1_score'], reverse=True)

    triggered = {
        'rules': [],
        'rule_ids': [],
        'confidences': []
    }
  
    for idx, (_, row) in enumerate(data.iterrows()):
        instance_preds = {}
        instance_scores = {}
        ruleset = []
        confidences = []
        for ant in archive:
            for rule in ant['ruleset']['rules']:
                # separate antecedent vs consequent
                antecedent = [(attr, val) for (attr, val) in rule['rule'] if attr not in labels]
                consequent = [(attr, val) for (attr, val) in rule['rule'] if attr in labels]
                # check if antecedent matches
                if all(row[attr] == val for (attr, val) in antecedent):
                    for i, (label, assigned) in enumerate(consequent):
                        instance_preds[label] = assigned
                        instance_scores[label] = rule['scores'][i][1]

                        # avoid duplicates in triggered rules
                        if rule['rule'] not in ruleset:
                            ruleset.append(rule['rule'])
                            confidences.append(rule['scores'][i][1] if assigned == '1' else (1 - np.int64(rule['scores'][i][1])))

        # fallback: assign majority for missing labels
        for label in labels:
            if label not in instance_preds:
                instance_preds[label] = '1' if priors.get(label) >= 0.5 else '0'
                instance_scores[label] = priors.get(label)

                # record triggered rules
                ruleset.append(f'{label} => Majority Class Fallback')
                confidences.append(priors.get(label) if instance_preds[label] == '1' else 1 - priors.get(label))
        
        triggered['rules'].append(ruleset)
        triggered['rule_ids'].append(f'rule{idx}')
        triggered['confidences'].append(np.mean(confidences) if confidences else 1.0)

        predictions[idx] = [instance_preds[label] for label in labels]
        scores[idx] = [instance_scores[label] for label in labels]

    return predictions, scores, triggered


def select_best(archive):
    """
    Selects the best rule from the archive based on the highest F1 score.
    """
    return max(archive, key=lambda ant: ant['f1_score'])


def select_closest(archive):
    """
    Selects the rule closest to the ideal point (1.0 for all fitness metrics).
    """
    return min(archive, key=lambda ant: sum((f - 1)**2 for f in ant['fitness'])**0.5)


def average_distance_to_ideal(ruleset):
    """
    Computes the average distance of a ruleset to the ideal point (1.0 for all fitness metrics).
    """
    return sum(sum((f - 1)**2 for f in ant['fitness'])**0.5 for ant in ruleset['ruleset']) / len(ruleset['ruleset'])


def predict_function(X, archive, archive_type, prediction_strat, labels, priors, task):
    """
    Predicts class labels for the given data using the rules or rulesets from the archive.
    """
    if len(archive) == 0:
        raise ValueError("No rules found in the archive. Run the algorithm first.")

    triggered = {
        'rules': [],
        'rule_ids': [],
        'confidences': []
    }
    triggered_rules = None
    scores = None

    y_preds = []

    if task == "multi":
        y_preds, scores, triggered_rules = predict_mlc(archive, X, labels, priors)
        return pd.DataFrame(y_preds, index=X.index, columns=labels), pd.DataFrame(scores, index=X.index, columns=labels), triggered_rules
    
    else:
        if archive_type == 'rules':
            archive = sorted(archive, key=lambda ant: ant['f1_score'], reverse=True)
    
            if prediction_strat == 'all':
                y_preds, triggered_rules = predict(X, archive)
    
            elif prediction_strat == 'best':
                best_ant = select_best(archive)
                y_preds, triggered_rules = predict(X, [best_ant])
    
            elif prediction_strat == 'reference':
                closest_ant = select_closest(archive)
                y_preds, triggered_rules = predict(X, [closest_ant])
    
    
        elif archive_type == 'rulesets':
    
            if prediction_strat == 'all':
                for idx, row in X.iterrows():
                    predicted = 'neg'
                    for ruleset in archive:
                        if any(matches_rule(row, ant['rule']) for ant in ruleset['ruleset']):
                            predicted = 'pos'
                            for ant in ruleset['ruleset']:
                                if matches_rule(row, ant['rule']):
                                    triggered['rules'].append(ant['rule'])
                                    triggered['rule_ids'].append(f'rule{idx}')
                                    triggered['confidences'].append(ant['fitness'][0])
                                    break
                            break
                    y_preds.append(predicted)
                    if predicted == 'neg':
                        triggered['rules'].append('No Match - Default to neg')
                        triggered['rule_ids'].append(f'rule{idx}')
                        triggered['confidences'].append(1.0)
                    triggered_rules = triggered
    
            elif prediction_strat == 'best':
                best_ruleset = select_best(archive)
                best_rules = sorted(best_ruleset['ruleset'], key=lambda ant: ant['f1_score'], reverse=True)
                y_preds, triggered_rules = predict(X, best_rules)
    
            elif prediction_strat == 'reference':
                closest_ruleset = min(archive, key=average_distance_to_ideal)
                best_rules = sorted(closest_ruleset['ruleset'], key=lambda ant: ant['f1_score'], reverse=True)
                y_preds, triggered_rules = predict(X, best_rules)

    return pd.Series(y_preds, index=X.index), scores, triggered_rules



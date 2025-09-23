import pandas as pd
import numpy as np
import math
import pprint
from .utils import get_label_probs, compute_entropy, calculate_terms_probs, select_term, covers_min_examples, assign_classes, assign_class_for_label, evaluate_rule
from .utils import build_contingency, compute_cramers_v, compute_v_threshold
from .metrics import accuracy, recall, precision, f1_score, hamming_loss, subset_accuracy, ranking_loss, one_error, coverage, average_precision


class MuLAM:

    def __init__(self, 
                 max_ants : int = 3000, 
                 max_uncovered : int = 5, 
                 min_covers : int = 5, 
                 alpha : int = 1,
                 beta : int = 1,
    ) -> None:
                
        self.max_ants = max_ants
        self.max_uncovered = max_uncovered
        self.min_covers = min_covers
        self.alpha = alpha
        self.beta = beta

        self.discovered_rules = []
        self.qualities = []
        self.majority = None


    def _initialize_heuristics(self, subset: pd.DataFrame, labels: list[str], terms: list[tuple]) -> dict:
        """
        Initialize heuristic values for each provided term.
        """
        heuristics = {}
        numerator_terms = {}
        num_classes = {label: subset[label].nunique() for label in labels}
    
        # iterate over provided terms
        for attr, val in terms:
            # compute IG for each label
            ig_sum = 0
            for label in labels:
                label_probs = get_label_probs(subset, attr, val, label)
                if not label_probs:
                    continue
    
                entropy = compute_entropy(label_probs)
                information_gain = math.log2(num_classes[label]) - entropy
                ig_sum += information_gain
    
            numerator_terms[(attr, val)] = ig_sum / len(labels) if labels else 0
    
        # normalize heuristics
        total = sum(numerator_terms.values())
        for term, value in numerator_terms.items():
            heuristics[term] = value / total if total != 0 else 0
    
        return heuristics
    

    def _initialize_pheromones(self, terms : list, labels: list[str]) -> dict:
        """
        Initialize pheromone matrix for each label attribute.
        """
        pheromones = {}

        for label in labels:
            pheromone = {term: 1.0 / len(terms) for term in terms}
            pheromones[label] = pheromone

        return pheromones
    

    def _construct_ruleset(self, subset : pd.DataFrame, labels : list[str], terms : list, pheromones : dict, heuristics : dict):
        """
        Construct a ruleset using the ant colony optimization approach.
        """
        ruleset = {'rules': []}
        rule_antecedent = []
        unpredicted_labels = labels.copy()
        unused_attrs = subset.drop(columns=labels).columns.tolist()

        while len(unused_attrs) > 0 and len(unpredicted_labels) > 0:
            terms_probs = calculate_terms_probs(terms, unpredicted_labels, pheromones, heuristics, unused_attrs, self.alpha, self.beta)
            selected_term = select_term(terms=terms, attrs=unused_attrs, p=terms_probs)

            if not covers_min_examples(data=subset, rule=rule_antecedent + [selected_term], threshold=self.min_covers):
                return ruleset, unpredicted_labels

            rule_antecedent += [selected_term]
            unused_attrs.remove(selected_term[0])

            rule_consequent = []

            covered_examples = subset.copy()
            for term in rule_antecedent:
                covered_examples = covered_examples[covered_examples[term[0]] == term[1]]

            labels_to_remove = []
            for label in unpredicted_labels:

                if len(covered_examples) == 0:
                    continue

                p_positive = (covered_examples[label] == 1).mean()
                if p_positive >= 0.90:
                    rule_consequent.append((label, 1))
                else:
                    rule_consequent.append((label, 0))

                labels_to_remove.append(label)
                #contingency = build_contingency(subset, covered_examples, label)

                #V = compute_cramers_v(contingency)
                #V_threshold = compute_v_threshold(subset, covered_examples, selected_term, label, contingency)
                # confidence 
                #confidence = len(covered_examples[covered_examples[label] == covered_examples[label].mode()[0]]) / len(covered_examples) if len(covered_examples) > 0 else 0
                #if confidence >= 0.75:
                #    majority_class = covered_examples[label].value_counts().idxmax()
                #    rule_consequent.append((label, majority_class))
                #    labels_to_remove.append(label)

                

            for label in labels_to_remove:
                unpredicted_labels.remove(label)
            
            if len(rule_consequent) > 0:
                
                complete_rule = rule_antecedent + rule_consequent

                qualities = []
                confidences = []
                for consequent in rule_consequent:
                    label_quality, label_confidence = evaluate_rule(rule=complete_rule, data=subset, label=consequent[0], labels=labels)
                    qualities.append(label_quality)
                    confidences.append(label_confidence)

                rule = {
                    'rule': complete_rule,
                    'quality': sum(qualities) / len(qualities) if qualities else 0,
                    'confidence': sum(confidences) / len(confidences) if confidences else 0
                }

                ruleset['rules'].append(rule)

        return ruleset, unpredicted_labels
    

    def _construct_complete_rule(self, subset : pd.DataFrame, labels : list[str], unpredicted_labels: list[str], terms : list[tuple], pheromones : dict, heuristics : dict):

        rule_antecedent = []
        rule_consequent = []
        unused_attrs = subset.drop(columns=labels).columns.tolist()

        while unused_attrs:
            terms_probs = calculate_terms_probs(terms, unpredicted_labels, pheromones, heuristics, unused_attrs, self.alpha, self.beta)

            selected_term = select_term(terms=terms, attrs=unused_attrs, p=terms_probs)

            if not covers_min_examples(data=subset, rule=rule_antecedent + [selected_term], threshold=self.min_covers):
                break

            rule_antecedent += [selected_term]
            unused_attrs.remove(selected_term[0])

        if len(rule_antecedent) > 0:
            rule_consequent = assign_classes(data=subset, rule_antecedent=rule_antecedent, labels=unpredicted_labels)
            
        return rule_antecedent, rule_consequent
    

    def _prune_rule(self, rule:list, subset: pd.DataFrame, label: str, labels: list[str]) -> list:
        """
        Prune the rule by removing terms until no further improvement is possible.
        """
        best_quality= evaluate_rule(rule=rule, data=subset, label=label, labels=labels)[0]
        pruned_rule = rule[:-1]

        while len(pruned_rule) > 1:

            best_term_to_remove = None
            best_improvement = best_quality

            for i, term in enumerate(pruned_rule):
                temp_rule = pruned_rule[:i] + pruned_rule[i+1:]
                rule_consequent = assign_class_for_label(data=subset, rule_antecedent=temp_rule, label=label)
                if rule_consequent[0] != label:
                    continue

                temp_rule += [rule_consequent]
                quality = evaluate_rule(rule=temp_rule, data=subset, label=label, labels=labels)[0]

                if quality > best_improvement:
                    best_improvement = quality
                    best_term_to_remove = term

            if best_term_to_remove:
                pruned_rule.remove(best_term_to_remove)
                best_quality = best_improvement
            else:
                break

        rule_consequent = assign_class_for_label(data=subset, rule_antecedent=pruned_rule, label=label)
        pruned_rule += [rule_consequent]

        quality, confidence = evaluate_rule(rule=pruned_rule, data=subset, label=label, labels=labels)
        return pruned_rule, quality, confidence
    

    def _drop_covered_examples(self, data: pd.DataFrame, ruleset: dict, labels: list[str]) -> pd.DataFrame:
        """
        Drop examples covered by the ruleset from the dataset.
        """
        uncovered_data = data.copy()

        for rule in ruleset['rules']:
            covered_examples = uncovered_data.copy()
            for term in rule['rule']:
                attr, val = term
                if attr in labels:
                    continue
                if attr not in uncovered_data.columns:
                    print(f"[WARN] Skipping term {term}, not a valid column")
                    continue
                
                covered_examples = covered_examples[covered_examples[attr] == val]

            uncovered_data = uncovered_data.drop(index=covered_examples.index).reset_index(drop=True)

        return uncovered_data
    

    def fit(self, data: pd.DataFrame, labels: list[str]) -> None:
        """
        Fit the MuLAM algorithm to the training data.
        """
        terms = [(col, val) for col in data.drop(columns=labels).columns for val in data[col].unique()]
        uncovered_data = data.copy()

        while len(uncovered_data) > self.max_uncovered:

            all_rulesets = []

            # init ant index
            ant = 0

            # init heuristics
            heuristics = self._initialize_heuristics(subset=uncovered_data, labels=labels, terms=terms)

            # init pheromones
            pheromones = self._initialize_pheromones(terms=terms, labels=labels)

            while ant < self.max_ants:

                ruleset, unpredicted_labels = self._construct_ruleset(subset=uncovered_data, labels=labels, terms=terms, pheromones=pheromones, heuristics=heuristics)

                
                if len(unpredicted_labels) > 0:
                    rule_antecedent, rule_consequent = self._construct_complete_rule(subset=uncovered_data, labels=labels, unpredicted_labels=unpredicted_labels, terms=terms, pheromones=pheromones, heuristics=heuristics)
                    
                    if rule_antecedent and rule_consequent:
                        for term in rule_consequent:
                            temp_rule = rule_antecedent + [term]

                            pruned_rule, quality, confidence = self._prune_rule(rule=temp_rule, subset=uncovered_data, label=term[0], labels=labels)

                            rule = {
                                'rule': pruned_rule,
                                'quality': quality,
                                'confidence': confidence
                            }

                            ruleset['rules'].append(rule)
                
                for rule in ruleset['rules']:
                    labels_in_rule = [term[0] for term in rule['rule'] if term[0] in labels]
                    
                    for label in labels_in_rule:
                        for term in terms:
                            if term in rule:
                                pheromones[label][term] += pheromones[label][term] * rule['quality']

                        pheromones_sum = sum(pheromones[label].values())
                        for term in terms:
                            pheromones[label][term] /= pheromones_sum

                ruleset['quality'] = sum(rule['quality'] for rule in ruleset['rules']) / len(ruleset['rules']) if ruleset['rules'] else 0

                all_rulesets.append(ruleset)

                ant += 1

            all_rulesets = sorted(all_rulesets, key=lambda x: x['quality'], reverse=True)
            best_ruleset = all_rulesets[0]
            self.discovered_rules.append(best_ruleset)
            uncovered_data = self._drop_covered_examples(data=uncovered_data, ruleset=best_ruleset, labels=labels)


    def predict(self, data: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict labels for the given data using the discovered rulesets.
        Returns a binary prediction matrix (n_samples x n_labels).
        """
        n_samples = len(data)
        n_labels = len(labels)
        predictions = np.zeros((n_samples, n_labels), dtype=int)
        scores = np.zeros((n_samples, n_labels), dtype=float)

        # sort rulesets by quality
        self.discovered_rules = sorted(self.discovered_rules, key=lambda x: x['quality'], reverse=True)

        # Majority fallback (per label)
        if self.majority is None:
            self.majority = {label: data[label].mode()[0] for label in labels}

        for idx, (_, row) in enumerate(data.iterrows()):
            instance_preds = {}
            instance_scores = {}
            for ruleset in self.discovered_rules:
                for rule in ruleset['rules']:
                    # sort rules by quality
                    ruleset['rules'] = sorted(ruleset['rules'], key=lambda x: x['quality'], reverse=True)
                    # separate antecedent vs consequent
                    antecedent = [(attr, val) for (attr, val) in rule['rule'] if attr not in labels]
                    consequent = [(attr, val) for (attr, val) in rule['rule'] if attr in labels]

                    # check if antecedent matches
                    if all(row[attr] == val for (attr, val) in antecedent):
                        for label, assigned in consequent:
                            instance_preds[label] = assigned
                            instance_scores[label] = max(instance_scores.get(label, 0), rule['confidence'])

            # fallback: assign majority for missing labels
            for label in labels:
                if label not in instance_preds:
                    instance_preds[label] = self.majority[label]
                    instance_scores[label] = 0.0

            predictions[idx] = [instance_preds[label] for label in labels]
            scores[idx] = [instance_scores[label] for label in labels]

        return predictions, scores
    

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
        """
        Evaluate the predictions using various multi-label metrics.
        """
        results = {}

        results['accuracy'] = accuracy(y_true, y_pred)
        results['recall'] = recall(y_true, y_pred)
        results['precision'] = precision(y_true, y_pred)
        results['f1_macro'] = f1_score(y_true, y_pred, 'macro')
        results['f1_micro'] = f1_score(y_true, y_pred, 'micro')
        results['hamming_loss'] = hamming_loss(y_true, y_pred)
        results['subset_accuracy'] = subset_accuracy(y_true, y_pred)
        results['ranking_loss'] = ranking_loss(y_true, scores)
        results['one_error'] = one_error(y_true, scores)
        results['coverage'] = coverage(y_true, scores)
        results['average_precision'] = average_precision(y_true, scores)

        return results

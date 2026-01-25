import pandas as pd
import numpy as np
import math
import time

from .utils import check_attributes_left, rule_covers_min_examples, select_term, calculate_terms_probs, compute_entropy, assign_class, evaluate_rule, plot_patero_front, update_EP
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

class AntMiner2:

    def __init__(self, 
                 max_ants : int = 3000, 
                 max_uncovered : int = 10, 
                 min_covers : int = 10, 
                 nb_converge : int = 10, 
                 alpha : int = 1,
                 beta : int = 1,
                 pruning : int = 1,
                 objs : list = ['specificity', 'sensitivity']
    ) -> None:
                
        self.max_ants = max_ants
        self.max_uncovered = max_uncovered
        self.min_covers = min_covers
        self.nb_converge = nb_converge 
        self.alpha = alpha
        self.beta = beta
        self.pruning = pruning
        self.objs = objs

        self.discovered_rules = []
        self.archive = []
        self.fitness_archive = []
        self.qualities = []
        self.majority = None

        self.training_history = {
            'train': [],
            'test': []
        }

    def _initialize_pheromones(self, terms : list) -> dict:
        """
        Initialize pheromone levels for each term.
        """
        pheromones = {term: 1.0 / len(terms) for term in terms}
        return pheromones
    

    def _get_class_probabilities(self, data : pd.DataFrame, attr_name : str, attr_value : tuple, class_label : str) -> list:
        """
        Calculate the class probabilities for a given attribute value.
        """
        subset = data[data[attr_name] == attr_value]
        total = len(subset)
        if total == 0:
            return []
        
        class_counts = subset[class_label].value_counts().tolist()
        probs = [count / total for count in class_counts]
        return probs
    

    def _initialize_heuristics(self, data : pd.DataFrame, class_name : str) -> dict:
        """
        Initialize heuristic values for each term.
        """

        num_classes = data[class_name].nunique()
        heuristics = {}
        numerator_terms = {}

        for attribute in data.columns[:-1]:
            for value in data[attribute].unique():
                probs = self._get_class_probabilities(data, attribute, value, class_name)

                if len(probs) > 0:
                    entropy = compute_entropy(probs)
                    term = (attribute, value)
                    numerator_terms[term] = math.log2(num_classes) - entropy

        total = sum(numerator_terms.values())
        for term, value in numerator_terms.items():
            heuristics[term] = value / total if total != 0 else 0

        return heuristics

    
    def _construct_rule(self, data : pd.DataFrame, all_terms : list, pheromone : dict, heuristic : dict) -> list:
        """
        Construct a rule using roulette wheel mechanism.
        """

        rule = []
        used_attrs = []

        all_atrs = set([term[0] for term in all_terms])

        while check_attributes_left(attrs=all_atrs, sub_attrs=used_attrs):

            probs = calculate_terms_probs(terms=all_terms, 
                                          pheromones=pheromone, 
                                          heuristics=heuristic, 
                                          used_attrs=used_attrs, 
                                          alpha=self.alpha, 
                                          beta=self.beta
                                          )

            selected_term = select_term(terms=all_terms, 
                                        used_attrs=used_attrs, 
                                        p=probs
                                        )

            if not rule_covers_min_examples(data=data, rule=rule + [selected_term], threshold=self.min_covers):
                break

            rule += [selected_term]
            used_attrs += [selected_term[0]]

        if len(rule) > 0:
            rule = assign_class(data=data, rule=rule)

        return rule


    def _prune_rule(self, rule : list, data : pd.DataFrame) -> list:
        """
        Prune the rule by removing terms until no further improvement is possible.
        """
        
        best_quality = evaluate_rule(rule=rule, data=data, label='class', objs=self.objs)[0]
        pruned_rule = rule[:-1]
        
        while len(pruned_rule) > 1:

            best_term_to_remove = None
            best_improvement = best_quality

            for i, term in enumerate(pruned_rule):
                temp_rule = pruned_rule[:i] + pruned_rule[i+1:]
                temp_rule = assign_class(data=data, rule=temp_rule)
                quality = evaluate_rule(rule=temp_rule, data=data, label='class', objs=self.objs)[0]

                if quality > best_improvement:
                    best_term_to_remove = term
                    best_improvement = quality

            if best_term_to_remove:
                pruned_rule.remove(best_term_to_remove)
                best_quality = best_improvement
            else:
                break

        pruned_rule = assign_class(data=data, rule=pruned_rule)

        return pruned_rule
    
    
    def _update_pheromones(self, pheromones : dict, rule : list, terms : list, quality : float) -> dict:
        """
        Update pheromone levels based on the quality of the rule.
        """

        for term in rule[:-1]:
            pheromones[term] += pheromones[term] * quality

        pheromones_sum = sum(pheromones.values())
        for term in terms:
                pheromones[term] = (pheromones[term] / pheromones_sum)

        return pheromones

    def _get_best_rule(self, rules : list, qualities : list, fitnesses: list) -> tuple:
        """
        Choose the best rule from a set of rules based on their quality.
        """
        best_rule = rules[0]
        best_quality = qualities[0]
        best_fitness = fitnesses[0]
        for i in range(1, len(rules)):
            if qualities[i] > best_quality:
                best_rule = rules[i]
                best_quality = qualities[i]
                best_fitness = fitnesses[i]

        return best_rule, best_quality, best_fitness
    
    
    def _drop_covered(self, best_rule : list, uncovered_data : pd.DataFrame) -> pd.DataFrame: 
        """
        Drop the instances covered by the best rule.
        """
        subset = uncovered_data.copy()
        for term in best_rule:
            subset = subset[subset[term[0]] == term[1]]

        uncovered_data = uncovered_data.drop(subset.index)
        uncovered_data = uncovered_data.reset_index(drop=True)

        return uncovered_data


    def fit(self, X, y, X_val, y_val):
        """
        Fit the AntMiner model to the training data.
        """
        data = X.copy()
        data['class'] = y

        all_terms = [(col, val) for col in X.columns for val in X[col].unique()]
        uncovered_data = data.copy()

        # compute heuristic values
        heuristics = self._initialize_heuristics(uncovered_data, 'class')

        while len(uncovered_data) > self.max_uncovered:

            if len(uncovered_data) <= self.max_uncovered:
                uncovered_data = data.copy()
            
            # initialize pheromone levels
            pheromones = self._initialize_pheromones(all_terms)

            ant = 1 # ant index 
            j = 1 # convergence test index

            all_rules = []
            qualities = []
            fitnesses = []

            prev_rule = ""
            
            while ant < self.max_ants and j < self.nb_converge:
                
                # construct the rule
                rule = self._construct_rule(uncovered_data, all_terms, pheromones, heuristics)
                rule_str = ""

                if len(rule) > 0:
                    # prune the rule
                    if self.pruning:
                        rule= self._prune_rule(rule, uncovered_data)
    
                    # evaluate the rule
                    quality, fitness = evaluate_rule(rule=rule, data=data, label='class', objs=self.objs)
    
                    # update pheromones
                    pheromones = self._update_pheromones(pheromones, rule, all_terms, quality)
    
                    # store the rules
                    all_rules.append(rule)
                    qualities.append(quality)
                    fitnesses.append(fitness)
                    
                    # rule to string
                    rule_str = " AND ".join(sorted([f"{term[0]} = {term[1]}" for term in rule[:-1]])) + f" THEN {rule[-1][1]}"

                # update convergence test
                if prev_rule == rule_str:
                    j += 1
                else:
                    j = 1
                
                prev_rule = rule_str
                ant += 1

            
            if len(all_rules) == 0:
                continue
            
            # choose the best rule
            best_rule, best_quality, best_fitness = self._get_best_rule(rules=all_rules, qualities=qualities, fitnesses=fitnesses)

            # archive the best rule
            self.discovered_rules.append(best_rule)
            self.qualities.append(best_quality)
            self.fitness_archive.append(best_fitness)

            # get non-dominated solutions
            self.archive = update_EP(
                rules=self.discovered_rules, 
                fitnesses=self.fitness_archive, 
                EP=self.archive
            )

            rule_str ="IF " + " AND ".join([f"({term[0]} = {term[1]})" for term in best_rule[:-1]]) + f" ==> (Class = {best_rule[-1][1]})"
            #print(f'Rule: {rule_str}, Quality: {best_quality}')

            # drop covered instances
            uncovered_data = self._drop_covered(best_rule, uncovered_data)

            # store convergence
            self.majority = uncovered_data['class'].mode()[0] if len(uncovered_data) > 0 else data['class'].mode()[0]
            self.store_convergence(X=X, y=y, X_val=X_val, y_val=y_val)

        self.majority = uncovered_data['class'].mode()[0] if len(uncovered_data) > 0 else data['class'].mode()[0]

    def store_convergence(self, X, y, X_val, y_val):
        tr_acc, tr_f1, tr_recall, tr_precision, tr_specificity = self.evaluate(X=X, y=y)
        val_acc, val_f1, val_recall, val_precision, val_specificity = self.evaluate(X=X_val, y=y_val)

        self.training_history['train'].append({
            'accuracy': tr_acc,
            'f1_score': tr_f1,
            'recall': tr_recall,
            'precision': tr_precision,
            'specificity': tr_specificity
        })
        self.training_history['test'].append({
            'accuracy': val_acc,
            'f1_score': val_f1,
            'recall': val_recall,
            'precision': val_precision,
            'specificity': val_specificity
        })


    def predict(self, X):
        """
        Predict the class for new instances based on the best rule in the archive.
        """
        if len(self.discovered_rules) == 0:
            raise ValueError("No rules found in the archive. Run the algorithm first.")
        
        # majority class in uncovered data

        y_preds = []
        for _, row in X.iterrows():
            predicted_class = None
            for rule in self.discovered_rules:
                if all(row[term[0]] == term[1] for term in rule[:-1]):
                    predicted_class = rule[-1][1]
                    break
            if predicted_class is None:
                predicted_class = self.majority
            y_preds.append(predicted_class)
        return pd.Series(y_preds, index=X.index)
    
    def pareto_predict(self, X):
        """
        Predict the class for new instances based on the Pareto front rules in the archive.
        """
        if len(self.archive) == 0:
            raise ValueError("No rules found in the archive. Run the algorithm first.")
        
        y_preds = []
        for _, row in X.iterrows():
            predicted_class = None
            for ant in self.archive:
                rule = ant['rule']
                if all(row[term[0]] == term[1] for term in rule[:-1]):
                    predicted_class = rule[-1][1]
                    break
            if predicted_class is None:
                predicted_class = self.majority
            y_preds.append(predicted_class)
        return pd.Series(y_preds, index=X.index)
    
    def get_rules(self):
        """
        Get the discovered rules.
        """
        return self.discovered_rules
    
    def get_term_rule_ratios(self):
        terms = 0
        for rule in self.discovered_rules:
            terms += len(rule) - 1
        return terms / len(self.discovered_rules) if self.discovered_rules else 0

    
    def evaluate(self, X, y):
        """
        Evaluate the ruleMiner model on the test data.
        """

        y_pred = self.pareto_predict(X)
        tn, fp, fn, tp = confusion_matrix(y_true=y, y_pred=y_pred, labels=['neg', 'pos']).ravel()

        accuracy = (tn + tp) / (tn + tp + fn + fp) if (tn + tp + fn + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return accuracy, f1, recall, precision, specificity

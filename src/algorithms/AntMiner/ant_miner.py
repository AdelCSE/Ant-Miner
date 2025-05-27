import pandas as pd
import math

from .utils import check_attributes_left, rule_covers_min_examples, select_term, calculate_terms_probs, compute_entropy, assign_class, evaluate_rule
from sklearn.metrics import f1_score, accuracy_score


class AntMiner:

    def __init__(self, 
                 max_ants : int = 3000, 
                 max_uncovered : int = 10, 
                 min_covers : int = 10, 
                 nb_converge : int = 10, 
                 alpha : int = 1,
                 beta : int = 1,
                 pruning : int = 1
    ) -> None:
                
        self.max_ants = max_ants
        self.max_uncovered = max_uncovered
        self.min_covers = min_covers
        self.nb_converge = nb_converge 
        self.alpha = alpha
        self.beta = beta
        self.pruning = pruning

        self.discovered_rules = []
        self.qualities = []


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

        while check_attributes_left(attrs=data.columns.tolist(), sub_attrs=used_attrs):

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

        rule = assign_class(data=data, rule=rule)

        return rule


    def _prune_rule(self, rule : list, data : pd.DataFrame) -> list:
        """
        Prune the rule by removing terms until no further improvement is possible.
        """
        
        best_quality = evaluate_rule(rule=rule, data=data)
        pruned_rule = rule[:-1]
        
        while len(pruned_rule) > 1:

            best_term_to_remove = None
            best_improvement = best_quality

            for i, term in enumerate(pruned_rule):
                temp_rule = pruned_rule[:i] + pruned_rule[i+1:]
                temp_rule = assign_class(data=data, rule=temp_rule)
                quality = evaluate_rule(rule=temp_rule, data=data)

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

    def _get_best_rule(self, rules : list, qualities : list) -> tuple:
        """
        Choose the best rule from a set of rules based on their quality.
        """
        best_rule = rules[0]
        best_quality = qualities[0]
        for i in range(1, len(rules)):
            if qualities[i] > best_quality:
                best_rule = rules[i]
                best_quality = qualities[i]

        return best_rule, best_quality
    
    
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


    def fit(self, X, y):
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

            # initialize pheromone levels
            pheromones = self._initialize_pheromones(all_terms)

            ant = 1 # ant index 
            j = 1 # convergence test index

            all_rules = []
            qualities = []

            prev_rule = ""
            
            while ant < self.max_ants and j < self.nb_converge:
                
                # construct the rule
                rule = self._construct_rule(uncovered_data, all_terms, pheromones, heuristics)

                # prune the rule
                if self.pruning:
                    rule= self._prune_rule(rule, uncovered_data)

                # evaluate the rule
                quality = evaluate_rule(rule=rule, data=data)

                # update pheromones
                pheromones = self._update_pheromones(pheromones, rule, all_terms, quality)

                # store the rules
                all_rules.append(rule)
                qualities.append(quality)
                
                # rule to string
                rule_str = " AND ".join(sorted([f"{term[0]} = {term[1]}" for term in rule[:-1]])) + f" THEN {rule[-1][1]}"

                # update convergence test
                if prev_rule == rule_str:
                    j += 1
                else:
                    j = 1
                
                prev_rule = rule_str
                ant += 1

            # choose the best rule
            best_rule, best_quality = self._get_best_rule(all_rules, qualities)

            # archive the best rule
            self.discovered_rules.append(best_rule)
            self.qualities.append(best_quality)

            rule_str ="IF " + " AND ".join([f"({term[0]} = {term[1]})" for term in best_rule[:-1]]) + f" ==> (Class = {best_rule[-1][1]})"
            print(f'Rule: {rule_str}, Quality: {best_quality}')

            # drop covered instances
            uncovered_data = self._drop_covered(best_rule, uncovered_data)


    def predict(self, X):
        """
        Predict the classes of new instances using discovered rules by order
        first rule that satisfy instance is applied
        """
        y_preds = []
        for _, row in X.iterrows():
            predicted_class = None
            for rule in self.discovered_rules:
                if all(row[term[0]] == term[1] for term in rule[:-1]):
                    predicted_class = rule[-1][1]
                    break
            y_preds.append(predicted_class)
        return pd.Series(y_preds, index=X.index)

    
    def evaluate(self, X, y):
        """
        Evaluate the AntMiner model on the test data.
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')

        return accuracy, f1

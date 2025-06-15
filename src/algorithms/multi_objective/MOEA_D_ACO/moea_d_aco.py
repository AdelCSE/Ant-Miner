import math
import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Import algorithm components
from .rules_initialization import initialize_rules
from .indicator import indicator_function
from .colony import create_colony
from .fitness import fitness_function
from .archive import update_EP
from .pheromones import update_pheromone
from .neighborhood import find_best_neighborhood_rule
from .patero_front import plot_patero_front
from .utils import get_class_probs, compute_entropy, assign_class, drop_covered


class MOEA_D_ACO():
    def __init__(self, population, neighbors, groups, max_iter, min_examples, max_uncovered, p, gamma, alpha, beta, delta, pruning):
        self.population = population
        self.neighbors = neighbors
        self.groups = groups
        self.max_iter = max_iter
        self.min_examples = min_examples
        self.max_uncovered = max_uncovered
        self.pruning = pruning

        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.gamma = gamma
        self.eps = 1 / (2 * population)

        self.lambda_weights = []
        self.lambda_groups = [[] for _ in range(groups)]
        self.group_ant = np.zeros(population, dtype=int)

        self.heuristics = None
        self.pheromones = None
        self.tau_max = 1
        self.tau_min = 1
        self.delta = delta * self.tau_max

        self.colony = {}
        self.neighborhoods = np.zeros((population, neighbors), dtype=int)
        self.replacing_solution = [[] for _ in range(population)]
        self.ARCHIVE = []


    def init_weights(self):
        """
        Initialize lambda weights and group them into clusters.
        """
        self.lambda_weights = np.array([[w, 1 - w] for w in np.linspace(0.0, 1.0, self.population)])
        csi_weights = np.array([[c, 1 - c] for c in np.linspace(0.0, 1.0, self.groups)])

        # Group lambda_weights into clusters
        for i, weight in enumerate(self.lambda_weights):
            dists = [cdist([weight], [csi]) for csi in csi_weights]
            group = int(np.argmin(dists))
            self.lambda_groups[group].append(weight)
            self.group_ant[i] = group
            

    def init_heuristics(self, data : pd.DataFrame, population : int, class_name : str):
        """
        Initialize heuristic information for each term in subproblem i.
        """
        num_classes = data[class_name].nunique()

        self.heuristics = {}
        for i in range(population):
            subproblem_heuristic = {}
            numerator_terms = {}
            for attribute in data.columns[:-1]:
                for value in data[attribute].unique():
                    probs = get_class_probs(data, attribute, value, class_name)

                    if len(probs) > 0:
                        entropy = compute_entropy(probs)
                        term = (attribute, value)
                        numerator_terms[term] = math.log2(num_classes) - entropy

            total = sum(numerator_terms.values())
            for term, value in numerator_terms.items():
                subproblem_heuristic[term] = value / total if total != 0 else 0
            
            self.heuristics[i] = subproblem_heuristic


    def init_pheromones(self, terms : list):
        """
        Initialize pheromone levels for each term in group k.
        """
        self.pheromones = {}
        for k in range(self.groups):
            pheromone = {term: 1.0 / len(terms) for term in terms}
            self.pheromones[k] = pheromone


    def init_neighborhood(self, population: int, neighbors: int):
        """
        Initialize neighborhoods for each ant in the colony.
        """
        self.neighborhoods = {}
        for i in range(population):
            lower = max(0, i - neighbors // 2)
            upper = min(population, lower + neighbors)
            self.neighborhoods[i] = np.arange(lower, upper)


    def _prune_rule(self, rule : list, data : pd.DataFrame) -> list:
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

        print(f'original rule: {rule}')
        print(f'pruned rule: {pruned_rule}')

        return pruned_rule
    


    def run(self, X, y):

        data = X.copy()
        data['class'] = y

        uncovered_data = data.copy()

        terms = [(col, val) for col in X.columns for val in X[col].unique()]

        # Initialize colony parameters
        self.init_weights()
        self.init_heuristics(data, self.population, 'class')
        self.init_pheromones(terms)

        # Initialize colony
        self.colony = initialize_rules(
            colony={}, data=data, attributes=X.columns.tolist(), terms=terms, 
            population=self.population, min_examples=self.min_examples
        )

        for t in tqdm(range(self.max_iter), desc="Running MOEA/D-ACO"):

            # Check if there is enough uncovered data
            if len(uncovered_data) <= self.max_uncovered:
                print(f"Early stopping at iteration {t} due to insufficient data.")
                break

            # Desirability matrix
            phi_matrix = {}
            for i in range(self.population):
                phi_matrix[i] = {}
                for term in terms:
                    indicator = indicator_function(rule=self.colony['ant'][i]['rule'], next_term=term)
                    g = self.group_ant[i]
                    tau_term = self.pheromones[g][term] + self.delta * indicator
                    phi_matrix[i][term] = (tau_term ** self.alpha) * (self.heuristics[i][term] ** self.beta)

            # Construct new colony
            self.colony = create_colony(
                data=uncovered_data, attributes=X.columns.tolist(), terms=terms, population=self.population, 
                gamma=self.gamma, phi=phi_matrix, min_examples=self.min_examples
            )

            # Prune rules
            if self.pruning:
                for i in range(self.population):
                    self.colony['ant'][i]['rule'] = self._prune_rule(
                        rule=self.colony['ant'][i]['rule'], data=data
                    )

            # Fitness evaluation
            for i in range(self.population):
                self.colony['ant'][i]['fitness'] = fitness_function(
                    data=data, rule=self.colony['ant'][i]['rule']
                )

            # Update EP
            ant_best_rule, self.ARCHIVE = update_EP(
                colony=self.colony, EP=self.ARCHIVE
            )

            # Update pheromones
            self.pheromones, self.tau_min, self.tau_max = update_pheromone(
                pheromones=self.pheromones, colony=self.colony,
                best_ants_indices=ant_best_rule, p=self.p, ant_groups=self.group_ant, 
                lambda_weights=self.lambda_weights, eps=self.eps
            )

            # Update neighborhood solutions
            for i in range(self.population):
                self.colony, self.replacing_solution = find_best_neighborhood_rule(
                    self.colony, data, i, self.neighborhoods,
                    self.replacing_solution, self.neighbors, self.lambda_weights
                )

            # Drop covered data
            for ant in ant_best_rule:
                uncovered_data = drop_covered(
                    best_rule=self.colony['ant'][ant]['rule'], data=uncovered_data
                )

        print('Archive:')
        pprint.pprint(self.ARCHIVE)

        # Plot the Pareto front
        plot_patero_front(archive=self.ARCHIVE)

    
    def predict(self, X):
        """
        Predict the class for new instances based on the best rule in the archive.
        """
        if len(self.ARCHIVE) == 0:
            raise ValueError("No rules found in the archive. Run the algorithm first.")

        y_preds = []
        for _, row in X.iterrows():
            predicted_class = None
            for ant in self.ARCHIVE:
                if all(row[term[0]] == term[1] for term in ant['rule'][:-1]):
                    predicted_class = ant['rule'][-1][1]
                    break
            y_preds.append(predicted_class)
        return pd.Series(y_preds, index=X.index)
    
    def evaluate(self, X, y):
        """
        Evaluate the performance of the discovered rules on a test set.
        """
        predictions = self.predict(X)
        accuracy = (predictions == y).mean()
        return accuracy

import math
import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

# Import algorithm components
from .colony import create_colony
from .pruning import prune_rule
from .fitness import fitness_function
from .archive import update_archive
from .pheromones import update_pheromone
from .neighborhood import find_best_neighborhood_rule
from .prediction import predict_function
from .patero_front import plot_patero_front
from .utils import get_class_probs, compute_entropy, drop_covered, remove_dominated_rules


class MOEA_D_AM():
    """
    Multi-Objective Evolutionary Algorithm based on Decomposition with Ant Colony Optimization (MOEA/D-ACO) 
    for rule discovery in classification tasks..

    Args:
        population (int): Number of ants in the colony.
        neighbors (int): Number of neighbors for each ant.
        groups (int): Number of groups for decomposition.
        max_iter (int): Maximum number of iterations.
        min_examples (int): Minimum number of examples required to cover by a rule.
        max_uncovered (int): Maximum number of uncovered examples allowed before stopping.
        p (float): Pheromone evaporation rate.
        gamma (float): Probability threshold for greedy selection.
        alpha (float): Weight for pheromone influence.
        beta (float): Weight for heuristic influence.
        delta (float): Pheromone update factor.
        pruning (int): Pruning strategy (0 for no pruning).
    
    Output:
        An instance of the MOEA_D_ACO class that can be used to run the algorithm
    """

    def __init__(self, 
                 population: int, 
                 neighbors: int, 
                 groups: int, 
                 max_iter: int, 
                 min_examples: int, 
                 max_uncovered: int, 
                 p: float, 
                 gamma: float, 
                 alpha: float, 
                 beta: float, 
                 delta: float, 
                 pruning: int,
                 archive_type: str,
                 rulesets: str,
                 random_state: int = None
    ) -> None:
        
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
        self.ant_groups = np.zeros(population, dtype=int)

        self.heuristics = None
        self.pheromones = None
        self.tau_max = 1
        self.tau_min = 1
        self.delta = delta * self.tau_max

        self.colony = {}
        self.ARCHIVE = []
        self.neighborhoods = np.zeros((population, neighbors), dtype=int)
        self.replacing_solution = [[] for _ in range(population)]

        self.random_state = random_state
        self.majority = None

        self.archive_type = archive_type
        self.rulesets = rulesets



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
            self.ant_groups[i] = group
            


    def init_heuristics(self, data : pd.DataFrame):
        """
        Initialize heuristic information for each term in subproblem i.
        """
        nb_classes = data['class'].nunique()

        self.heuristics = {}
        numerator_terms = {}

        for attribute in data.columns[:-1]:
            for value in data[attribute].unique():
                probs = get_class_probs(data, attribute, value, 'class')

                if len(probs) > 0:
                    entropy = compute_entropy(probs)
                    term = (attribute, value)
                    numerator_terms[term] = math.log2(nb_classes) - entropy
                    
        total = sum(numerator_terms.values())
        for term, value in numerator_terms.items():
            self.heuristics[term] = value / total if total != 0 else 0
        


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
    


    def run(self, X, y, labels):

        data = X.copy()
        data['class'] = y
        positive_class = labels[0]

        terms = [(col, val) for col in X.columns for val in X[col].unique()]
        uncovered_data = data.copy()

        # Initialize colony parameters
        self.init_weights()
        self.init_heuristics(data)
        self.init_pheromones(terms)

        for t in tqdm(range(self.max_iter), desc="Running MOEA/D-AM"):

            # Check if there is enough uncovered data
            if len(uncovered_data) <= self.max_uncovered:
                #print(f"Early stopping at iteration {t} due to insufficient data.")
                break

            # Desirability matrix
            phi_matrix = {}
            for i in range(self.population):
                phi_matrix[i] = {}
                for term in terms:
                    g = self.ant_groups[i]
                    phi_matrix[i][term] = (self.pheromones[g][term] ** self.alpha) * (self.heuristics[term] ** self.beta)

            # Construct new colony
            self.colony = create_colony(
                data=uncovered_data, attributes=X.columns.tolist(), terms=terms, population=self.population, 
                gamma=self.gamma, phi=phi_matrix, min_examples=self.min_examples, random_state=self.random_state
            )

            # Prune rules
            if self.pruning:
                for i in range(self.population):
                    self.colony['ants'][i]['rule'] = prune_rule(
                        data=data, rule=self.colony['ants'][i]['rule']
                    )

            # Fitness evaluation
            for i in range(self.population):
                self.colony['ants'][i]['fitness'], self.colony['ants'][i]['f1_score'] = fitness_function(
                    data=data, rule=self.colony['ants'][i]['rule']
                )

            # Update archive
            best_ants_indices, self.ARCHIVE = update_archive(
                colony=self.colony, archive=self.ARCHIVE, positive_class=positive_class, rulesets=self.rulesets,
            )

            # Update pheromones
            self.pheromones, self.tau_min, self.tau_max = update_pheromone(
                pheromones=self.pheromones, colony=self.colony,
                best_ants_indices=best_ants_indices, p=self.p, ant_groups=self.ant_groups, 
                lambda_weights=self.lambda_weights, eps=self.eps
            )

            
            # Update neighborhood solutions
            for i in range(self.population):
                self.colony, self.replacing_solution = find_best_neighborhood_rule(
                    self.colony, data, i, self.neighborhoods, positive_class,
                    self.replacing_solution, self.neighbors, self.lambda_weights
                )

            
            for i in best_ants_indices:
                # Drop covered examples from uncovered_data
                uncovered_data = drop_covered(
                    best_rule=self.colony['ants'][i]['rule'],
                    data=uncovered_data,
                )
            
        #self.ARCHIVE = remove_dominated_rules(self.ARCHIVE)
        self.majority = uncovered_data['class'].mode()[0] if len(uncovered_data) > 0 else None

        if self.rulesets == 'subproblem':
            self.ARCHIVE = [ant for ant in self.ARCHIVE if ant['f1_score'] > 0]


        #pprint.pprint(self.ARCHIVE)
        
    
    def get_term_rule_ratios(self):

        if self.archive_type == 'rules':
            terms = 0
            for ant in self.ARCHIVE:
                terms += len(ant['rule']) - 1

            return terms / len(self.ARCHIVE) if self.ARCHIVE else 0
        
        elif self.archive_type == 'rulesets':
            terms = 0
            rules = 0
            for ruleset in self.ARCHIVE:
                for ant in ruleset['ruleset']:
                    terms += len(ant['rule']) - 1
                    rules += 1

            return terms / rules if rules > 0 else 0

    
    def predict(self, X):
        """
        Predict the class for new instances based on the best rule in the archive.
        """
        if len(self.ARCHIVE) == 0:
            raise ValueError("No rules found in the archive. Run the algorithm first.")
        
        
        if self.majority is None:
            classes = [ant['rule'][-1][1] for ant in self.ARCHIVE]
            self.majority = pd.Series(classes).mode()[0] if len(classes) > 0 else None

        y_preds = []
        for _, row in X.iterrows():
            predicted_class = None
            for ant in self.ARCHIVE:
                if all(row[term[0]] == term[1] for term in ant['rule'][:-1]):
                    predicted_class = ant['rule'][-1][1]
                    break
            if predicted_class == None:
                predicted_class = self.majority

            y_preds.append(predicted_class)
        return pd.Series(y_preds, index=X.index)
    


    def partial_predict(self, X, labels):
        """
        Predict the class for new instances based on the best rule in the archive.
        This method is similar to predict but returns a DataFrame with probabilities.
        """
        if len(self.ARCHIVE) == 0:
            raise ValueError("No rules found in the archive. Run the algorithm first.")
        

        y_preds = []
        for _, row in X.iterrows():
            predicted_class = None
            for ant in self.ARCHIVE:
                if all(row[term[0]] == term[1] for term in ant['rule'][:-1]):
                    predicted_class = labels[0]
                    break
            if predicted_class is None:
                predicted_class = labels[1]

            y_preds.append(predicted_class)
        
        return pd.Series(y_preds, index=X.index)



    def evaluate(self, X, y, labels, prediction_strat):
        """
        Evaluate the performance of the discovered rules on a test set.
        """
        y_pred = predict_function(
            X=X, archive=self.ARCHIVE, labels=labels, archive_type=self.archive_type, prediction_strat=prediction_strat
        )

        accuracy = accuracy_score(y, y_pred)
        f1_score_v = f1_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        precision = precision_score(y, y_pred, average='weighted')

        return accuracy, f1_score_v, recall, precision
    

    def get_archive(self):
        """
        Get the archive of discovered rules.
        """
        return self.ARCHIVE
    

    def get_ant_colony(self):
        """
        Get the ant colony.
        """
        return self.colony
    



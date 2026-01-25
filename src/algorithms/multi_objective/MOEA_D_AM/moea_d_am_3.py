import math
import pprint
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# Import algorithm components
from .colony2 import create_colony
from .pruning import prune_rule
from .fitness import fitness_function
from .archive import update_archive
from .pheromones import update_pheromone
from .neighborhood import find_best_neighborhood_rule
from .prediction import predict_function
from .utils import get_class_probs, compute_entropy, drop_covered, get_term_rule_ratio, update_reference_point
from .hypervolume import hypervolume

class MOEA_D_AM_3():
    """
    Multi-Objective Evolutionary Algorithm based on Decomposition with Ant Colony Optimization (MOEA/D-ACO) 
    for rule discovery in Single-Label Classification (strictly 'pos'/'neg').

    Args:
        ants_per_subproblem (int): Number of candidate ants to generate for each subproblem before selection.
    """

    def __init__(self,
                 task: str,
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
                 pruning: int,
                 decomposition: str,
                 archive_type: str,
                 prediction_strat: str,
                 rulesets: str,
                 objs: list,
                 random_state: int = None,
                 ants_per_subproblem: int = 50 
    ) -> None:
        
        self.task = task

        self.population = population
        self.neighbors = neighbors
        self.groups = groups
        self.ants_per_subproblem = ants_per_subproblem

        self.max_iter = max_iter
        self.min_examples = min_examples
        self.max_uncovered = max_uncovered
        self.pruning = pruning

        self.objs = objs

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

        self.colony = {}
        self.ARCHIVE = []
        self.neighborhoods = np.zeros((population, neighbors), dtype=int)
        self.replacing_solution = [[] for _ in range(population)]

        self.random_state = random_state
        self.majority = None

        self.decomposition = decomposition
        self.archive_type = archive_type
        self.rulesets = rulesets
        self.prediction_strat = prediction_strat

        self.reference_point = np.array([0.0, 0.0])

        self.priors = {}

        self.hypervolume_history = []
        self.all_points = []

        self.training_history = {
            'train': [],
            'test': []
        }

    def init_weights(self):
        """Initialize lambda weights and group them into clusters."""
        self.lambda_weights = np.array([[w, 1 - w] for w in np.linspace(0.0, 1.0, self.population)])
        csi_weights = np.array([[c, 1 - c] for c in np.linspace(0.0, 1.0, self.groups)])

        for i, weight in enumerate(self.lambda_weights):
            dists = [cdist([weight], [csi]) for csi in csi_weights]
            group = int(np.argmin(dists))
            self.lambda_groups[group].append(weight)
            self.ant_groups[i] = group

    def init_heuristics(self, data : pd.DataFrame, labels: list[str], terms: list[tuple]):
        """Initialize heuristic information (Information Gain)."""
        self.heuristics = {}
        numerator_terms = {}
        num_classes = {label: data[label].nunique() for label in labels}

        for attr, val in terms:
            ig_sum = 0
            for label in labels:
                probs = get_class_probs(data, attr, val, label)
                if not probs: continue
                entropy = compute_entropy(probs)
                information_gain = math.log2(num_classes[label]) - entropy
                ig_sum += information_gain 
            
            numerator_terms[(attr, val)] = ig_sum / len(labels) if labels else 0

        total = sum(numerator_terms.values())
        for term, value in numerator_terms.items():
            self.heuristics[term] = value / total if total != 0 else 0

    def init_pheromones(self, task: str, terms : list, labels: list[str] = []):
        """Initialize pheromones."""
        self.pheromones = {}
        pheromone = {term: 1.0 / len(terms) for term in terms}
        for k in range(self.groups):
            self.pheromones[k] = pheromone

    def init_neighborhood(self, population: int, neighbors: int):
        """Initialize circular neighborhoods."""
        self.neighborhoods = {}
        half = neighbors // 2
        for i in range(population):
            indices = [(i + j) % population for j in range(-half, half + 1)]
            if neighbors % 2 == 0:
                indices = indices[:-1]
            self.neighborhoods[i] = np.array(indices)

    def run(self, data: pd.DataFrame, labels: list[str], Val_data: pd.DataFrame = None):

        # Calculate priors for majority voting fallback
        self.priors = {
            'class': data['class'].value_counts(normalize=True).idxmax()
        }

        terms = [(col, val) for col in data.drop(columns=labels).columns for val in data[col].unique()]
        uncovered_data = data.copy()

        # Initialize colony parameters
        self.init_weights()
        self.init_heuristics(data, labels=labels, terms=terms)
        self.init_pheromones(self.task, terms, labels)
        self.init_neighborhood(self.population, self.neighbors)
        
        for _ in tqdm(range(self.max_iter), desc="Training"):

            iteration_points = []

            if len(uncovered_data) <= self.max_uncovered:
                break

            # 1. Calculate Desirability (Phi) for all subproblems
            phi_matrix = {}
            for i in range(self.population):
                phi_matrix[i] = {}
                g = self.ant_groups[i]
                for term in terms:
                    phi_matrix[i][term] = (self.pheromones[g][term] ** self.alpha) * (self.heuristics[term] ** self.beta)

            # 2. Generate Candidate Pool (Returns List of Lists)
            candidate_colony = create_colony(
                data=uncovered_data, attributes=data.drop(columns=labels).columns.tolist(),
                labels=labels, terms=terms, population=self.population, gamma=self.gamma, phi=phi_matrix, 
                min_examples=self.min_examples, random_state=self.random_state,
                ants_per_subproblem=self.ants_per_subproblem 
            )

            # 3. Tournament Selection (Filter Candidates)
            self.colony = {'ants': []}

            for i in range(self.population):
                candidates = candidate_colony['ants'][i]
                
                best_ant = None
                best_score = -1.0

                for ant in candidates:
                    # A. Pruning
                    if self.pruning and len(ant['rule']) > 2:
                        ant['rule'] = prune_rule(
                            data=uncovered_data, ant=ant, task=self.task, labels=['class'], objs=self.objs
                        )

                    # B. Fitness
                    fitness, score = fitness_function(
                        data=data, ant=ant, labels=labels, task=self.task, objs=self.objs
                    )
                    
                    ant['fitness'] = fitness
                    ant['score'] = score

                    # C. Select Best for Subproblem
                    if score > best_score:
                        best_score = score
                        best_ant = ant
                
                # Add winner to official colony
                self.colony['ants'].append(best_ant)
                iteration_points.append(best_ant['fitness'])

            # 4. Update Archive
            best_ants_indices, self.ARCHIVE = update_archive(
                colony=self.colony, archive=self.ARCHIVE, rulesets=self.rulesets,
            )

            # Update reference point
            if self.decomposition == 'tchebycheff':
                self.reference_point = update_reference_point(self.reference_point, self.colony, self.task, maximize=True)

            # 5. Update Pheromones
            self.pheromones, self.tau_min, self.tau_max = update_pheromone(
                colony=self.colony, pheromones=self.pheromones, best_ants=best_ants_indices, 
                p=self.p, ant_groups=self.ant_groups, lambda_weights=self.lambda_weights, 
                eps=self.eps, decomposition=self.decomposition, reference=self.reference_point,
                labels=labels, task=self.task
            )

            # 6. Neighborhood Update
            for i in range(self.population):
                self.colony, self.replacing_solution = find_best_neighborhood_rule(
                    colony=self.colony, data=data, ant_index=i, neighborhood=self.neighborhoods,
                    rep_list=self.replacing_solution, neighbors=self.neighbors, weights=self.lambda_weights,
                    decomposition=self.decomposition, reference=self.reference_point, labels=labels,
                    task=self.task, objs=self.objs
                )

            # 7. Drop Covered Examples (Sequential Covering)
            if len(best_ants_indices) > 0:
                best_ant = self.get_best_ant(self.colony, best_ants_indices)
                subset = drop_covered(
                    best_ant=best_ant,
                    data=uncovered_data,
                    task=self.task
                )
                uncovered_data.drop(index=subset.index, inplace=True)
            
            # Store history
            self.store_convergence(X=data.drop(columns=labels), y=data[labels[0]], X_val=Val_data.drop(columns=labels), y_val=Val_data[labels[0]])
            
        self.all_points.extend(iteration_points)
        
        # Filter archive
        if self.rulesets == 'subproblem':
            self.ARCHIVE = [ant for ant in self.ARCHIVE if ant['score'] > 0]
        
    def get_best_ant(self, colony: dict, best_ants_indices: list[int]) -> dict:
        """Get the best ant based on global score (Accuracy/F1)."""
        best_ant = None
        best_score = -1.0
        for i in best_ants_indices:
            if colony['ants'][i]['score'] > best_score:
                best_score = colony['ants'][i]['score']
                best_ant = colony['ants'][i]
        return best_ant

    def store_convergence(self, X, y, X_val, y_val):
        ypred, _, _ = self.predict(X, archive=self.ARCHIVE, archive_type=self.archive_type, prediction_strat=self.prediction_strat, labels=['class'], priors=self.priors)
        y_pred_val, _, _ = self.predict(X_val, archive=self.ARCHIVE, archive_type=self.archive_type, prediction_strat=self.prediction_strat, labels=['class'], priors=self.priors)

        tr_acc, tr_f1, tr_recall, tr_precision, tr_specificity, _, _, _ = self.evaluate_slc(y_true=y, y_pred=ypred)
        val_acc, val_f1, val_recall, val_precision, val_specificity, _, _, _ = self.evaluate_slc(y_true=y_val, y_pred=y_pred_val)

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

    def predict(self, X: pd.DataFrame, archive: dict, archive_type: str, prediction_strat: str, labels: list[str], priors: dict):
        return predict_function(X=X, archive=archive, archive_type=archive_type, prediction_strat=prediction_strat, labels=labels, priors=priors, task='single')
    
    def evaluate_slc(self, y_true, y_pred):
        """
        Evaluate performance using strict 'neg'/'pos' labels.
        """
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=['neg', 'pos']).ravel()

        accuracy = (tn + tp) / (tn + tp + fn + fp) if (tn + tp + fn + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_score_v = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        nb_rules = len(self.ARCHIVE)
        tr_ratio = get_term_rule_ratio(self.ARCHIVE, self.archive_type, ['class'], self.task)
        hypervolume = self.get_hypervolume()

        return accuracy, f1_score_v, recall, precision, specificity, nb_rules, tr_ratio, hypervolume

    def get_hypervolume(self):
        if len(self.ARCHIVE) == 0: return 0.0
        front = np.array([ant['fitness'] for ant in self.ARCHIVE])
        reference_point = np.array([0.0, 0.0])
        return hypervolume(front, reference_point)
    
    def get_hv_history(self): 
        return self.hypervolume_history
    
    def get_all_points(self): 
        return self.all_points
    
    def get_archive(self): 
        return self.ARCHIVE
    
    def get_priors(self): 
        return self.priors
    
    def get_ant_colony(self): 
        return self.colony
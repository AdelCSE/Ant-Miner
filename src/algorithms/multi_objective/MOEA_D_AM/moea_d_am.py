import math
import pprint
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

# Import algorithm components
from .colony import create_colony
from .pruning import prune_rule
from .fitness import fitness_function
from .archive import update_archive
from .pheromones import update_pheromone
from .neighborhood import find_best_neighborhood_rule
from .prediction import predict_function
from .utils import get_class_probs, compute_entropy, drop_covered, get_term_rule_ratio, update_reference_point
from .hypervolume import hypervolume
from .patero_front import plot_patero_front
from .metrics import accuracy, recall, precision, f1_measure, hamming_loss, subset_accuracy, ranking_loss, one_error, coverage, average_precision


class MOEA_D_AM():
    """
    Multi-Objective Evolutionary Algorithm based on Decomposition with Ant Colony Optimization (MOEA/D-ACO) 
    for rule discovery in classification tasks.

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
        archive_type (str): Type of archive to maintain ('rules' or 'rulesets').
        rulesets (str): Strategy for handling rulesets ('subproblem' or 'iteration').
        ruleset_size (int): Size of each ruleset in the archive (default is 2).
        random_state (int, optional): Random seed for reproducibility.
    
    Output:
        An instance of the MOEA_D_ACO class that can be used to run the algorithm
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
                 random_state: int = None
    ) -> None:
        
        # Task type: 'single-label' or 'multi-label'
        self.task = task

        self.population = population
        self.neighbors = neighbors
        self.groups = groups

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



    def init_heuristics(self, data : pd.DataFrame, labels: list[str], terms: list[tuple]):
        """
        Initialize heuristic information for each term in subproblem i.
        """
        self.heuristics = {}
        numerator_terms = {}
        num_classes = {label: data[label].nunique() for label in labels}

        for attr, val in terms:
            ig_sum = 0
            for label in labels:
                probs = get_class_probs(data, attr, val, label)
                if not probs:
                    continue

                entropy = compute_entropy(probs)
                information_gain = math.log2(num_classes[label]) - entropy
                ig_sum += information_gain 
            
            numerator_terms[(attr, val)] = ig_sum / len(labels) if labels else 0

        total = sum(numerator_terms.values())
        for term, value in numerator_terms.items():
            self.heuristics[term] = value / total if total != 0 else 0



    def init_pheromones(self, task: str, terms : list, labels: list[str] = []):
        """
        Initialize pheromone levels for each term in group k.
        """
        self.pheromones = {}
        pheromone = {term: 1.0 / len(terms) for term in terms}

        if task == 'single':
            for k in range(self.groups):
                self.pheromones[k] = pheromone
        else:
            for k in range(self.groups):
                self.pheromones[k] = {}
                for label in labels:
                    self.pheromones[k][label] = pheromone

            

    def init_neighborhood(self, population: int, neighbors: int):
        """
        Initialize neighborhoods for each ant in the colony (circular neighborhood).
        """
        self.neighborhoods = {}
        half = neighbors // 2

        for i in range(population):
            indices = [(i + j) % population for j in range(-half, half + 1)]
            if neighbors % 2 == 0:
                indices = indices[:-1]
            self.neighborhoods[i] = np.array(indices)


    def run(self, data: pd.DataFrame, labels: list[str], Val_data: pd.DataFrame = None):

        if self.task == 'multi':
            self.priors = {
                label: data[label].value_counts(normalize=True).get('1', 0)
                for label in labels
            }
        else:
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

        
        
        t = 0
        start_time = time.time()
        while True:
            if time.time() - start_time > 5:
                #print(f"Stopping at iteration {t} due to time limit.")
                break
        
        #for t in tqdm(range(self.max_iter), desc="Running MOEA/D-AM"):

            iteration_points = []

            # Check if there is enough uncovered data
            if len(uncovered_data) <= self.max_uncovered:
                print(f"Early stopping at iteration {t} due to insufficient data.")
                break

            # Desirability matrix
            phi_matrix = {}
            for i in range(self.population):
                phi_matrix[i] = {}
                for term in terms:
                    g = self.ant_groups[i]
                    if self.task == 'multi':
                        phi_matrix[i][term] = 0
                        for label in labels:
                            phi_matrix[i][term] += (self.pheromones[g][label][term] ** self.alpha) * (self.heuristics[term] ** self.beta)
                        phi_matrix[i][term] /= len(labels)
                    else:
                        phi_matrix[i][term] = (self.pheromones[g][term] ** self.alpha) * (self.heuristics[term] ** self.beta)

            # Construct new colony
            self.colony = create_colony(
                task=self.task, data=uncovered_data, attributes=data.drop(columns=labels).columns.tolist(),
                labels=labels, terms=terms, population=self.population, gamma=self.gamma, phi=phi_matrix, 
                min_examples=self.min_examples, random_state=self.random_state
            )

            # Prune rules
            if self.pruning:
                for i in range(self.population):
                    if self.task == 'single':
                        if len(self.colony['ants'][i]['rule']) > 2:
                            self.colony['ants'][i]['rule'] = prune_rule(
                                data=uncovered_data, ant=self.colony['ants'][i], task=self.task, labels=['class'], objs=self.objs
                            )
                    else:
                        if any(len(rule['rule']) > 2 for rule in self.colony['ants'][i]['ruleset']['rules']):
                            self.colony['ants'][i]['ruleset'] = prune_rule(
                                data=uncovered_data, ant=self.colony['ants'][i], task=self.task, labels=labels, objs=self.objs
                            )

            # Fitness evaluation
            for i in range(self.population):
                fitness, f1_score = fitness_function(
                    data=data, ant=self.colony['ants'][i], labels=labels, task=self.task, objs=self.objs
                )
                if self.task == 'single':
                    self.colony['ants'][i]['fitness'] = fitness
                    self.colony['ants'][i]['f1_score'] = f1_score
                else:
                    self.colony['ants'][i]['ruleset']['fitness'] = fitness
                    self.colony['ants'][i]['ruleset']['f1_score'] = f1_score

                iteration_points.append(fitness)
            
            # Update archive
            best_ants_indices, self.ARCHIVE = update_archive(
                colony=self.colony, archive=self.ARCHIVE, rulesets=self.rulesets,
            )

            # Update reference point
            if self.decomposition == 'tchebycheff':
                self.reference_point = update_reference_point(self.reference_point, self.colony, self.task, maximize=True)

            # Update pheromones
            self.pheromones, self.tau_min, self.tau_max = update_pheromone(
                colony=self.colony, pheromones=self.pheromones, best_ants=best_ants_indices, 
                p=self.p, ant_groups=self.ant_groups, lambda_weights=self.lambda_weights, 
                eps=self.eps, decomposition=self.decomposition, reference=self.reference_point,
                labels=labels,task=self.task
            )

            # Update neighborhood solutions
            for i in range(self.population):
                self.colony, self.replacing_solution = find_best_neighborhood_rule(
                    colony=self.colony, data=data, ant_index=i, neighborhood=self.neighborhoods,
                    rep_list=self.replacing_solution, neighbors=self.neighbors, weights=self.lambda_weights,
                    decomposition=self.decomposition, reference=self.reference_point, labels=labels,
                    task=self.task, objs=self.objs
                )

            """
            for i in best_ants_indices:
                # Drop covered examples from uncovered_data
                uncovered_data = drop_covered(
                    best_ant=self.colony['ants'][i],
                    data=uncovered_data,
                    task=self.task
                )
            """
            # store convergence
            if self.task == 'single':
                self.store_convergence(X=data.drop(columns=labels), y=data[labels[0]], X_val=Val_data.drop(columns=labels), y_val=Val_data[labels[0]])

            # store hypervolume history
            hv_t = self.get_hypervolume()
            self.hypervolume_history.append(hv_t)

        self.all_points.extend(iteration_points)

        #self.ARCHIVE = remove_dominated_rules(self.ARCHIVE)

        if self.rulesets == 'subproblem':
            self.ARCHIVE = [ant for ant in self.ARCHIVE if ant['f1_score'] > 0]
        
        #pprint.pprint(self.ARCHIVE)


    def store_convergence(self, X, y, X_val, y_val):
        ypred, _, _ = self.predict(X, archive=self.ARCHIVE, archive_type=self.archive_type, prediction_strat=self.prediction_strat, labels=['class'], priors=self.priors, task='single')
        y_pred_val, _, _ = self.predict(X_val, archive=self.ARCHIVE, archive_type=self.archive_type, prediction_strat=self.prediction_strat, labels=['class'], priors=self.priors, task='single')

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


    def predict(self, X: pd.DataFrame, archive: dict, archive_type: str, prediction_strat: str, labels: list[str], priors: dict, task: str):
        """
        Predict the class for new instances based on the discovered rules.
        """
        if self.task == 'single':
            return predict_function(X=X, archive=archive, archive_type=archive_type, prediction_strat=prediction_strat, labels=labels, priors=priors, task=task)
        elif self.task == 'multi':
            return predict_function(X=X, archive=archive, archive_type=None, prediction_strat=None, labels=labels, priors=priors, task=task)
        else:
            raise ValueError("Unsupported task type. Use 'single' or 'multi'!")

    
    def predict_slc(self, X):
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
    

    def predict_mlc(self, data: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict the classes for new instances based on the discovered rulesets.
        """
        n_samples = len(data)
        n_labels = len(labels)
        predictions = np.zeros((n_samples, n_labels), dtype=int)
        scores = np.zeros((n_samples, n_labels), dtype=float)

        # sort rulesets by quality
        self.ARCHIVE = sorted(self.ARCHIVE, key=lambda x: x['ruleset']['f1_score'], reverse=True)

        for idx, (_, row) in enumerate(data.iterrows()):
            instance_preds = {}
            instance_scores = {}
            for ant in self.ARCHIVE:
                for rule in ant['ruleset']['rules']:

                    # separate antecedent vs consequent
                    antecedent = [(attr, val) for (attr, val) in rule['rule'] if attr not in labels]
                    consequent = [(attr, val) for (attr, val) in rule['rule'] if attr in labels]

                    # check if antecedent matches
                    if all(row[attr] == val for (attr, val) in antecedent):
                        for i, (label, assigned) in enumerate(consequent):
                            instance_preds[label] = assigned
                            instance_scores[label] = rule['scores'][i][1]

            
            # fallback: assign majority for missing labels
            for label in labels:
                if label not in instance_preds:
                    instance_preds[label] = 1 if self.priors.get(label) >= 0.5 else 0
                    instance_scores[label] = self.priors.get(label)

            predictions[idx] = [instance_preds[label] for label in labels]
            scores[idx] = [instance_scores[label] for label in labels]

        return predictions, scores


    def evaluate_slc(self, y_true, y_pred):
        """
        Evaluate the performance of the discovered rules on a test set.
        """
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=['neg', 'pos']).ravel()

        accuracy = (tn + tp) / (tn + tp + fn + fp) if (tn + tp + fn + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_score_v = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        nb_rules = len(self.ARCHIVE) if self.archive_type == 'rules' else sum(len(ant['ruleset']) for ant in self.ARCHIVE)
        tr_ratio = get_term_rule_ratio(self.ARCHIVE, self.archive_type, ['class'], self.task)
        hypervolume = self.get_hypervolume()

        return accuracy, f1_score_v, recall, precision, specificity, nb_rules, tr_ratio, hypervolume


    def evaluate_mlc(self, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], scores: np.ndarray) -> dict:
        """
        Evaluate multi-label classification performance using various metrics.
        """
        # to str
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        results = {}

        results['acc'] = accuracy(y_true, y_pred)
        results['recall'] = recall(y_true, y_pred, 'weighted')
        results['precision'] = precision(y_true, y_pred, 'weighted')
        results['f1_score'] = f1_measure(y_true, y_pred, 'weighted')
        results['f1_macro'] = f1_measure(y_true, y_pred, 'macro')
        results['f1_micro'] = f1_measure(y_true, y_pred, 'micro')
        results['hamming_loss'] = hamming_loss(y_true, y_pred)
        results['subset_acc'] = subset_accuracy(y_true, y_pred)
        results['ranking_loss'] = ranking_loss(y_true, scores)
        results['coverage'] = coverage(y_true, scores)
        results['avg_precision'] = average_precision(y_true, scores)
        results['hypervolume'] = self.get_hypervolume()
        results['nb_rulesets'] = len(self.ARCHIVE)
        results['term_rule_ratio'] = get_term_rule_ratio(self.ARCHIVE, self.archive_type, labels, self.task)

        return results


    def get_hypervolume(self):
        """
        Get the hypervolume of the discovered rules.
        """
        if len(self.ARCHIVE) == 0:
            return 0.0


        front = np.array([ant['fitness'] if self.task == 'single' else ant['ruleset']['fitness'] for ant in self.ARCHIVE])
        reference_point = np.array([0.0, 0.0])

        return hypervolume(front, reference_point)
    

    def get_hv_history(self):
        """
        Get the history of hypervolume values over iterations.
        """
        return self.hypervolume_history
    

    def get_all_points(self):
        """
        Get all points (objectives) explored during the optimization.
        """
        return self.all_points
    

    def get_archive(self):
        """
        Get the archive of discovered rules.
        """
        return self.ARCHIVE
    
    def get_priors(self):
        """
        Get the class priors.
        """
        return self.priors
    

    def get_ant_colony(self):
        """
        Get the ant colony.
        """
        return self.colony
    



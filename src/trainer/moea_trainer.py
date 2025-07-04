import pandas as pd
import dotenv
import time
import sys
import os
import pprint
from argparse import ArgumentParser
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

sys.path.append("../..")
from src.algorithms.multi_objective.MOEA_D_AM import MOEA_D_AM

env = dotenv.find_dotenv()

@dataclass
class Args:
    """
    Arguments for the MOEA/D-AM algorithm.
    """
    population : int
    neighbors : int
    groups : int
    min_examples : int
    max_uncovered : int
    max_iter : int
    gamma : float
    delta : float
    alpha : int
    beta : int
    p : float
    pruning : int
    archive_type : str
    rulesets : str
    prediction_strat : str
    cross_val : bool
    folds : int
    random_state : int


def main(args: Args) -> None:
    """
    Main function to run the MOEA/D-AM algorithm.
    """

    # Get the environment variables
    DATA_DIR = dotenv.get_key(env, 'DATA_DIR')
    RESULTS_DIR = dotenv.get_key(env, 'RESULTS_DIR')

    # Load the dataset
    dataframe = pd.read_csv(DATA_DIR)

    X = dataframe.drop('class', axis=1)
    y = dataframe['class']
    y = y.astype(str)
    labels = ['True', 'False']

    results = pd.DataFrame(columns=['fold', 'accuracy', 'f1_score', 'recall', 'precision', 'nb_rules', 'term_rule_ratio', 'time'])
    
    if args.cross_val:

        print(f'Starting cross-validation with {args.folds} folds...')
    
        sets = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        for k, (train_index, test_index) in enumerate(sets.split(X, y)):
    
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
            moea_d_aco = MOEA_D_AM(
                population=args.population,
                neighbors=args.neighbors,
                groups=args.groups,
                min_examples=args.min_examples,
                max_uncovered=args.max_uncovered,
                max_iter=args.max_iter,
                p=args.p,
                gamma=args.gamma,
                alpha=args.alpha,
                beta=args.beta,
                delta=args.delta,
                pruning=args.pruning,
                archive_type=args.archive_type,
                rulesets=args.rulesets,
                random_state=args.random_state
            )
    
            start_time = time.time()
            moea_d_aco.run(X=X_train, y=y_train, labels=labels)
            end_time = time.time()
        
            acc, f1, recall, precision = moea_d_aco.evaluate(X_test, y_test, labels, args.prediction_strat)
    
            nb_rules = len(moea_d_aco.ARCHIVE)
            term_rule_ratio = moea_d_aco.get_term_rule_ratios()
            time_taken = end_time - start_time
    
            results.loc[k] = [k + 1, acc, f1, recall, precision, nb_rules, term_rule_ratio, time_taken]
    
            print(f'Fold {k+1} completed in {time_taken:.2f} seconds [Accuracy: {acc:.2f}, F1 Score: {f1:.2f}, Recall: {recall:.2f}, Precision: {precision:.2f}, Rules: {nb_rules}, Term/Rule Ratio: {term_rule_ratio:.2f}]')

        dataset = DATA_DIR.split('/')[-1].split('.')[0]
        pruning_sfx = '_pruning' if args.pruning else '_no_pruning'

        if not os.path.exists(f"{RESULTS_DIR}/MOEA_D_AM/CV"):
            os.makedirs(f"{RESULTS_DIR}/MOEA_D_AM/CV")

        results.to_csv(f"{RESULTS_DIR}/MOEA_D_AM/CV/{dataset}{pruning_sfx}.csv", index=False)
    
        print(f"\nCross-validation completed within {results['time'].sum():.2f} seconds. {results['time'].mean():.2f} seconds per fold.")
        print(f"Average Accuracy: {results['accuracy'].mean() * 100:.2f} ± {results['accuracy'].std() * 100:.2f}")
        print(f"Average F1 Score: {results['f1_score'].mean() * 100:.2f} ± {results['f1_score'].std() * 100:.2f}")
        print(f"Average Recall: {results['recall'].mean() * 100:.2f} ± {results['recall'].std() * 100:.2f}")
        print(f"Average Precision: {results['precision'].mean() * 100:.2f} ± {results['precision'].std() * 100:.2f}")
        print(f"Average Number of Rules: {results['nb_rules'].mean():.2f} ± {results['nb_rules'].std():.2f}")
        print(f"Average Term/Rule Ratio: {results['term_rule_ratio'].mean():.2f} ± {results['term_rule_ratio'].std():.2f}")

    else:

        print('Starting training without cross-validation...')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        moea_d_aco = MOEA_D_AM(
            population=args.population,
            neighbors=args.neighbors,
            groups=args.groups,
            min_examples=args.min_examples,
            max_uncovered=args.max_uncovered,
            max_iter=args.max_iter,
            p=args.p,
            gamma=args.gamma,
            alpha=args.alpha,
            beta=args.beta,
            delta=args.delta,
            pruning=args.pruning,
            archive_type=args.archive_type,
            rulesets=args.rulesets,
            random_state=args.random_state
        )

        start_time = time.time()
        moea_d_aco.run(X=X_train, y=y_train, labels=labels)
        end_time = time.time()

        acc, f1, recall, precision = moea_d_aco.evaluate(X_test, y_test, labels, args.prediction_strat)
        nb_rules = len(moea_d_aco.ARCHIVE)
        term_rule_ratio = moea_d_aco.get_term_rule_ratios()
        time_taken = end_time - start_time

        results.loc[0] = [1, acc, f1, recall, precision, nb_rules, term_rule_ratio, time_taken]

        dataset = DATA_DIR.split('/')[-1].split('.')[0]
        pruning_sfx = '_pruning' if args.pruning else '_no_pruning'

        if not os.path.exists(f"{RESULTS_DIR}/MOEA_D_AM/ALL"):
            os.makedirs(f"{RESULTS_DIR}/MOEA_D_AM/ALL")

        results.to_csv(f"{RESULTS_DIR}/MOEA_D_AM/ALL/{dataset}{pruning_sfx}.csv", index=False)
        
        print(f'Training completed in {time_taken:.2f} seconds [Accuracy: {acc * 100:.2f}, F1 Score: {f1 * 100:.2f}, Recall: {recall * 100:.2f}, Precision: {precision * 100:.2f}, Rules: {nb_rules}, Term/Rule Ratio: {term_rule_ratio:.2f}]')


if __name__ == "__main__":

    parser = ArgumentParser(description="AntMiner Algorithm")

    parser.add_argument("--population", type=int, default=100, help="Ant population size")
    parser.add_argument("--neighbors", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--groups", type=int, default=5, help="Number of ant groups")
    parser.add_argument("--min-examples", type=int, default=10, help="Minimum number of examples to be covered by a rule")
    parser.add_argument("--max-uncovered", type=int, default=10, help="Maximum number of uncovered data")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum number of iterations")
    parser.add_argument("--gamma", type=float, default=0.9, help="Term selection probability threshold")
    parser.add_argument("--delta", type=float, default=0.5, help="Delta parameter for information matrix update")
    parser.add_argument("--alpha", type=int, default=1, help="Pheromones influence parameter")
    parser.add_argument("--beta", type=int, default=1, help="Heuristics influence parameter")
    parser.add_argument("--p", type=float, default=0.9, help="Evaporation rate of pheromones")
    parser.add_argument("--pruning", type=int, default=0, help="Pruning parameter")
    parser.add_argument("--archive-type", type=str, default="rulesets", choices=["rules", "rulesets"], help="Structure of archive")
    parser.add_argument("--rulesets", type=str, default='subproblem', choices=[None, 'iteration', 'subproblem'], help="The approach used to induce rulesets")
    parser.add_argument("--prediction-strat", type=str, default="all", choices=["all", "best", "reference", "voting"], help="Prediction strategy to use")
    parser.add_argument("--cross-val", type=bool, default=True, help="Enable cross-validation")
    parser.add_argument("--folds", type=int, default=10, help="Number of folds for cross-validation")
    parser.add_argument("--random-state", type=int, default=None, help="Random state for reproducibility")

    args = parser.parse_args()
    main(args)



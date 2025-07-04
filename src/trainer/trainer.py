import pandas as pd
import dotenv
import time
import sys
from sklearn.model_selection import StratifiedKFold

from argparse import ArgumentParser
from dataclasses import dataclass

sys.path.append("../..")
from src.algorithms.AntMiner import AntMiner


env = dotenv.find_dotenv()

@dataclass
class Args:
    """
    Arguments for the AntMiner algorithm.
    """
    max_ants : int
    max_uncovered : int
    min_covers : int
    nb_converge : int
    alpha : int
    beta : int
    pruning : int
    num_folds : int


def main(args: Args) -> None:
    """
    Main function to run the AntMiner algorithm.
    """
    # Load the dataset
    DATA_DIR = dotenv.get_key(env, 'DATA_DIR')

    dataframe = pd.read_csv(DATA_DIR)

    X = dataframe.drop('class', axis=1)
    y = dataframe['class']

    results = pd.DataFrame(columns=['fold', 'accuracy', 'f1_score', 'recall', 'precision', 'nb_rules', 'term_rule_ratio', 'time'])

    print(f'Starting cross-validation with {args.num_folds} folds...')

    sets = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    for k, (train_index, test_index) in enumerate(sets.split(X, y)):
        print(f'\n - Fold {k+1} / {args.num_folds}:')

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        ant_miner = AntMiner(
            max_ants=args.max_ants,
            max_uncovered=args.max_uncovered,
            min_covers=args.min_covers,
            nb_converge=args.nb_converge,
            alpha=args.alpha,
            beta=args.beta,
            pruning=args.pruning
        )

        start_time = time.time()
        ant_miner.fit(X_train, y_train)
        end_time = time.time()

        accuracy, f1, recall, precision= ant_miner.evaluate(X_test, y_test)

        nb_rules = len(ant_miner.discovered_rules)
        term_rule_ratio = ant_miner.get_term_rule_ratios()
        time_taken = end_time - start_time

        results.loc[k] = [k + 1, accuracy, f1, recall, precision, nb_rules, term_rule_ratio, time_taken]

        print(f'Fold {k+1} completed in {time_taken:.2f} seconds [Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}, Recall: {recall:.2f}, Precision: {precision:.2f}, Rules: {nb_rules}, Term/Rule Ratio: {term_rule_ratio:.2f}]')

    # Save results to CSV
    results.to_csv('/home/adel/Documents/Code/Ant-Miner/results/AntMiner/lg_no_pruning_all.csv', index=False)

    print(f"\nCross-validation completed within {results['time'].sum():.2f} seconds. {results['time'].mean():.2f} seconds per fold.")
    print(f"Average Accuracy: {results['accuracy'].mean() * 100:.2f} ± {results['accuracy'].std() * 100:.2f}")
    print(f"Average F1 Score: {results['f1_score'].mean() * 100:.2f} ± {results['f1_score'].std() * 100:.2f}")
    print(f"Average Recall: {results['recall'].mean() * 100:.2f} ± {results['recall'].std() * 100:.2f}")
    print(f"Average Precision: {results['precision'].mean() * 100:.2f} ± {results['precision'].std() * 100:.2f}")
    print(f"Average Number of Rules: {results['nb_rules'].mean():.2f} ± {results['nb_rules'].std():.2f}")
    print(f"Average Term/Rule Ratio: {results['term_rule_ratio'].mean():.2f} ± {results['term_rule_ratio'].std():.2f}")


if __name__ == "__main__":

    parser = ArgumentParser(description="AntMiner Algorithm")

    parser.add_argument("--max-ants", type=int, default=3000, help="Number of ants")
    parser.add_argument("--max-uncovered", type=int, default=10, help="Maximum number of uncovered training examples to stop the algorithm")
    parser.add_argument("--min-covers", type=int, default=10, help="Minimum number of examples to be covered by a rule")
    parser.add_argument("--nb-converge", type=int, default=10, help="Number of rules used to test convergence of the ants")
    parser.add_argument("--alpha", type=int, default=1, help="Alpha parameter for pheromone importance")
    parser.add_argument("--beta", type=int, default=1, help="Beta parameter for heuristic importance")
    parser.add_argument("--pruning", type=int, default=0, help="Enable rule pruning")
    parser.add_argument("--num-folds", type=int, default=10, help="Number of folds for cross-validation")

    args = parser.parse_args()
    main(args)



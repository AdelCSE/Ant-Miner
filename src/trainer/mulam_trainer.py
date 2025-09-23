import pandas as pd
import dotenv
import time
import sys
from sklearn.model_selection import StratifiedKFold

from argparse import ArgumentParser
from dataclasses import dataclass

sys.path.append("../..")
from src.algorithms.MuLAM import MuLAM


env = dotenv.find_dotenv()

@dataclass
class Args:
    """
    Arguments for the MuLAM algorithm.
    """
    max_ants : int
    max_uncovered : int
    min_covers : int
    alpha : int
    beta : int
    num_folds : int


def main(args: Args) -> None:
    """
    Main function to run the MuLAM algorithm.
    """
    # Load the dataset
    DATA_DIR = dotenv.get_key(env, 'MLC_DATA_DIR')

    data = pd.read_csv(DATA_DIR)

    labels = [col for col in data.columns if 'label' in col]

    results = pd.DataFrame(columns=['fold', 'accuracy', 'f1_score', 'recall', 'precision', 'nb_rules', 'term_rule_ratio', 'time'])

    print(f'Starting cross-validation with {args.num_folds} folds...')

    sets = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    for k, (train_index, test_index) in enumerate(sets.split(X, y)):
        print(f'\n - Fold {k+1} / {args.num_folds}:')

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        ant_miner = MuLAM(
            max_ants=args.max_ants,
            max_uncovered=args.max_uncovered,
            min_covers=args.min_covers,
            alpha=args.alpha,
            beta=args.beta,
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

if __name__ == "__main__":

    parser = ArgumentParser(description="MuLAM Algorithm")

    parser.add_argument("--max-ants", type=int, default=3000, help="Number of ants")
    parser.add_argument("--max-uncovered", type=int, default=10, help="Maximum number of uncovered training examples to stop the algorithm")
    parser.add_argument("--min-covers", type=int, default=10, help="Minimum number of examples to be covered by a rule")
    parser.add_argument("--alpha", type=int, default=1, help="Alpha parameter for pheromone importance")
    parser.add_argument("--beta", type=int, default=1, help="Beta parameter for heuristic importance")
    parser.add_argument("--num-folds", type=int, default=10, help="Number of folds for cross-validation")

    args = parser.parse_args()
    main(args)



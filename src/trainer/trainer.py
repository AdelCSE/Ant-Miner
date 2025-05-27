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

    Total_time = 0

    accuracy_list = []
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

        accuracy, f1score = ant_miner.evaluate(X_test, y_test)

        Total_time += end_time - start_time

        print(f'Fold {k+1} completed in {end_time - start_time:.2f} seconds [Accuracy: {accuracy:.2f}, F1-Score: {f1score:.2f}]')
        accuracy_list.append(accuracy)

    print(f'\nTotal time for {args.num_folds} folds: {Total_time:.2f} seconds')
    print(f'Average accuracy: {sum(accuracy_list) / len(accuracy_list):.2f} ± {pd.Series(accuracy_list).std():.2f}')


if __name__ == "__main__":

    parser = ArgumentParser(description="AntMiner Algorithm")

    parser.add_argument("--max-ants", type=int, default=3000, help="Number of ants")
    parser.add_argument("--max-uncovered", type=int, default=10, help="Maximum number of uncovered training examples to stop the algorithm")
    parser.add_argument("--min-covers", type=int, default=10, help="Minimum number of examples to be covered by a rule")
    parser.add_argument("--nb-converge", type=int, default=10, help="Number of rules used to test convergence of the ants")
    parser.add_argument("--alpha", type=int, default=1, help="Alpha parameter for pheromone importance")
    parser.add_argument("--beta", type=int, default=1, help="Beta parameter for heuristic importance")
    parser.add_argument("--pruning", type=int, default=1, help="Enable rule pruning")
    parser.add_argument("--num-folds", type=int, default=10, help="Number of folds for cross-validation")

    args = parser.parse_args()
    main(args)



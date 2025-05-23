import pandas as pd
import dotenv
import time
import sys

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


def main(args: Args) -> None:
    """
    Main function to run the AntMiner algorithm.
    """
    # Load the dataset
    DATA_DIR = dotenv.get_key(env, 'DATA_DIR')

    dataframe = pd.read_csv(DATA_DIR)
    X = dataframe.drop('class', axis=1)
    y = dataframe['class']

    # Initialize the AntMiner algorithm
    ant_miner = AntMiner(
        max_ants=args.max_ants,
        max_uncovered=args.max_uncovered,
        min_covers=args.min_covers,
        nb_converge=args.nb_converge,
        alpha=args.alpha,
        beta=args.beta
    )

    # Start the timer
    start_time = time.time()

    # Run the AntMiner algorithm
    ant_miner.fit(X=X, y=y)

    # Stop the timer
    end_time = time.time()

    print(f'Training completed successfully within {end_time - start_time:.2f} seconds.')


if __name__ == "__main__":

    parser = ArgumentParser(description="AntMiner Algorithm")

    parser.add_argument("--max_ants", type=int, default=3000, help="Number of ants")
    parser.add_argument("--max_uncovered", type=int, default=10, help="Maximum number of uncovered training examples to stop the algorithm")
    parser.add_argument("--min_covers", type=int, default=10, help="Minimum number of examples to be covered by a rule")
    parser.add_argument("--nb_converge", type=int, default=10, help="Number of rules used to test convergence of the ants")
    parser.add_argument("--alpha", type=int, default=1, help="Alpha parameter for pheromone importance")
    parser.add_argument("--beta", type=int, default=1, help="Beta parameter for heuristic importance")

    args = parser.parse_args()
    main(args)



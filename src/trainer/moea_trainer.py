import pandas as pd
import dotenv
import time
import sys

from argparse import ArgumentParser
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

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


def main(args: Args) -> None:
    """
    Main function to run the MOEA/D-AM algorithm.
    """
    # Load the dataset
    DATA_DIR = dotenv.get_key(env, 'DATA_DIR')

    dataframe = pd.read_csv(DATA_DIR)

    X = dataframe.drop('class', axis=1)
    y = dataframe['class']
    y = y.astype(str)

    # split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize the MOEA/D-AM algorithm
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
        pruning=args.pruning
    )

    print("Starting MOEA/D-AM algorithm with the following parameters:\n")
    print(f"Population size: {args.population} - Neighbors: {args.neighbors} - Groups: {args.groups} - Minimum examples to cover: {args.min_examples} - Max iterations: {args.max_iter} - P: {args.p} - Gamma: {args.gamma} - Alpha: {args.alpha} - Beta: {args.beta} - Delta: {args.delta} - Pruning: {args.pruning}\n")
    print('Train set shape:', X_train.shape)

    start_time = time.time()
    moea_d_aco.run(X_train, y_train)
    end_time = time.time()

    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    acc, f1_score, recall, precision, auc = moea_d_aco.evaluate(X_test, y_test)
    print(f"Accuracy: {acc:.4f}, F1 Score: {f1_score:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, AUC: {auc:.4f}")

if __name__ == "__main__":

    parser = ArgumentParser(description="AntMiner Algorithm")

    parser.add_argument("--population", type=int, default=20, help="Ant population size")
    parser.add_argument("--neighbors", type=int, default=5, help="Number of neighbors")
    parser.add_argument("--groups", type=int, default=2, help="Number of ant groups")
    parser.add_argument("--min-examples", type=int, default=10, help="Minimum number of examples to be covered by a rule")
    parser.add_argument("--max-uncovered", type=int, default=10, help="Maximum number of uncovered data")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum number of iterations")
    parser.add_argument("--gamma", type=float, default=0.9, help="Term selection probability threshold")
    parser.add_argument("--delta", type=float, default=0.5, help="Delta parameter for information matrix update")
    parser.add_argument("--alpha", type=int, default=1, help="Pheromones influence parameter")
    parser.add_argument("--beta", type=int, default=1, help="Heuristics influence parameter")
    parser.add_argument("--p", type=float, default=0.9, help="Evaporation rate of pheromones")
    parser.add_argument("--pruning", type=int, default=0, help="Pruning parameter")

    args = parser.parse_args()
    main(args)



import pandas as pd
import dotenv
import time
import sys
import json
from sklearn.model_selection import StratifiedKFold

from argparse import ArgumentParser
from dataclasses import dataclass

sys.path.append("../..")
from src.algorithms.AntMiner import AntMiner2


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
    objs : list
    dataset : str
    runs : int = 1


def main(args: Args) -> None:
    """
    Main function to run the AntMiner algorithm.
    """
    # Load the dataset
    DATA_DIR = dotenv.get_key(env, 'SLC_DATA_DIR')
    SAVE_DIR = dotenv.get_key(env, 'SLC_RESULTS_DIR')
    MODELS_DIR = dotenv.get_key(env, 'SLC_MODELS_DIR')

    dataframe = pd.read_csv(DATA_DIR + f'/{args.dataset}.csv', dtype=str)

    X = dataframe.drop('class', axis=1)
    y = dataframe['class']

    results = pd.DataFrame(columns=['run', 'fold', 'split', 'accuracy', 'f1_score', 'recall', 'precision', 'specificity', 'nb_rules', 'term_rule_ratio', 'time'])
    model = {}

    sets = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)

    for run_id in range(1, args.runs + 1):
        print(f'\n=== Run {run_id} / {args.runs} ===')
        for k, (train_index, test_index) in enumerate(sets.split(X, y)):
            #print(f'\n - Fold {k+1} / {args.num_folds}:')
    
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
            ant_miner = AntMiner2(
                max_ants=args.max_ants,
                max_uncovered=args.max_uncovered,
                min_covers=args.min_covers,
                nb_converge=args.nb_converge,
                alpha=args.alpha,
                beta=args.beta,
                pruning=args.pruning,
                objs=args.objs
            )
    
            start_time = time.time()
            ant_miner.fit(X=X_train, y=y_train, X_val=X_test, y_val=y_test)
            end_time = time.time()
    
            tr_acc, tr_f1, tr_rec, tr_prec, tr_spec = ant_miner.evaluate(X_train, y_train)
            ts_acc, ts_f1, ts_rec, ts_prec, ts_spec = ant_miner.evaluate(X_test, y_test)
    
            nb_rules = len(ant_miner.discovered_rules)
            term_rule_ratio = ant_miner.get_term_rule_ratios()
            time_taken = end_time - start_time
    
            results.loc[len(results)] = [run_id, k + 1, 'train', tr_acc, tr_f1, tr_rec, tr_prec, tr_spec, nb_rules, term_rule_ratio, time_taken]
            results.loc[len(results)] = [run_id, k + 1, 'test', ts_acc, ts_f1, ts_rec, ts_prec, ts_spec, nb_rules, term_rule_ratio, time_taken]
            
            fold_archive = ant_miner.discovered_rules
            fold_history = ant_miner.training_history
            model[f'run_{run_id}'] = model.get(f'run_{run_id}', {})
            model[f'run_{run_id}'][f'fold_{k+1}'] = {
                'archive': fold_archive,
                'history': fold_history
            }

    objs_str = '_'.join([obj[:4] for obj in args.objs])
    results.to_csv(SAVE_DIR + f'/AM/{args.dataset}_{objs_str}.csv', index=False)
    with open(MODELS_DIR + f'/AM/{args.dataset}_{objs_str}.json', 'w') as f:
        json.dump(model, f, indent=4)

    print(f"\nAverage results over {args.runs} runs:")
    print(results.groupby('split').mean())

if __name__ == "__main__":

    parser = ArgumentParser(description="AntMiner Algorithm")

    parser.add_argument("--max-ants", type=int, default=3000, help="Number of ants")
    parser.add_argument("--max-uncovered", type=int, default=10, help="Maximum number of uncovered training examples to stop the algorithm")
    parser.add_argument("--min-covers", type=int, default=10, help="Minimum number of examples to be covered by a rule")
    parser.add_argument("--nb-converge", type=int, default=10, help="Number of rules used to test convergence of the ants")
    parser.add_argument("--alpha", type=int, default=1, help="Alpha parameter for pheromone importance")
    parser.add_argument("--beta", type=int, default=1, help="Beta parameter for heuristic importance")
    parser.add_argument("--pruning", type=int, default=0, help="Enable rule pruning")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--objs", nargs='+', type=str, default=['specificity', 'sensitivity'], help="Fitness objectives for multi-objective optimization")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs to average results over")

    args = parser.parse_args()
    main(args)



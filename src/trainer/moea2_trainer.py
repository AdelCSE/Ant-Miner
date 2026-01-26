import pandas as pd
import dotenv
import time
import sys
import os
import json
import numpy as np

from argparse import ArgumentParser
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, StratifiedKFold

sys.path.append("../..")
from src.algorithms.multi_objective.MOEA_D_AM import MOEA_D_AM_3

env = dotenv.find_dotenv()

@dataclass
class Args:
    population : int
    neighbors : int
    groups : int
    ants : int
    min_examples : int
    max_uncovered : int
    max_iter : int
    gamma : float
    alpha : int
    beta : int
    p : float
    pruning : int

    decomposition : str
    archive_type : str
    rulesets : str
    prediction_strat : str
    drop_covered : bool

    cross_val : bool
    folds : int
    random_state : int
    runs : int

    objs : list
    dataset : str


def run_once(args: Args, X, y, labels, run_id: int, archive: dict, objs: list):
    """
    Executes one run of MOEA/D-AM for Single-Label Classification.
    """
    
    # Initialize results DataFrame with only SLC metrics
    results = pd.DataFrame(columns=[
        'run', 'fold', 'split', 'accuracy', 'f1_score', 'recall', 'precision', 'specificity',
        'nb_rules', 'term_rule_ratio', 'hypervolume', 'time'
    ])

    # 1. Cross-Validation Mode
    if args.cross_val:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)

        for k, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Combine for the algorithm (it expects a dataframe with the class column)
            train_data = pd.concat([X_train, y_train], axis=1).astype(str)
            test_data = pd.concat([X_test, y_test], axis=1).astype(str)

            # Instantiate Algorithm
            moea = MOEA_D_AM_3(
                task='single',
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
                pruning=args.pruning,
                decomposition=args.decomposition,
                archive_type=args.archive_type,
                prediction_strat=args.prediction_strat,
                rulesets=args.rulesets,
                drop_covered=args.drop_covered,
                objs=objs,
                random_state=args.random_state,
                ants_per_subproblem=args.ants
            )

            # Run Training
            start_time = time.time()
            moea.run(data=train_data, labels=labels, Val_data=test_data)
            end_time = time.time()
            fold_time = end_time - start_time

            # Retrieve Artifacts
            priors = moea.get_priors()
            fold_archive = moea.get_archive()
            
            # Store Model/History
            archive[f"run_{run_id}"] = archive.get(f"run_{run_id}", {})
            archive[f"run_{run_id}"][f"fold_{k+1}"] = {
                'archive': fold_archive,
                'history': moea.training_history,
                'priors': priors
            }

            # --- Evaluate on Training Set ---
            y_pred_train, _, _ = moea.predict(X=train_data, archive=fold_archive, archive_type=args.archive_type, 
                                      prediction_strat=args.prediction_strat, labels=labels, priors=priors)
            tr_metrics = moea.evaluate_slc(y_true=y_train.astype(str), y_pred=y_pred_train)
            
            results.loc[len(results)] = [
                run_id, k + 1, 'train', 
                tr_metrics[0]*100, tr_metrics[1]*100, tr_metrics[2]*100, tr_metrics[3]*100, tr_metrics[4]*100,
                tr_metrics[5], tr_metrics[6], tr_metrics[7], fold_time
            ]

            # --- Evaluate on Test Set ---
            y_pred_test, _, _ = moea.predict(X=test_data, archive=fold_archive, archive_type=args.archive_type, 
                                     prediction_strat=args.prediction_strat, labels=labels, priors=priors)
            ts_metrics = moea.evaluate_slc(y_true=y_test.astype(str), y_pred=y_pred_test)

            results.loc[len(results)] = [
                run_id, k + 1, 'test', 
                ts_metrics[0]*100, ts_metrics[1]*100, ts_metrics[2]*100, ts_metrics[3]*100, ts_metrics[4]*100,
                ts_metrics[5], ts_metrics[6], ts_metrics[7], fold_time
            ]

            print(f"Fold {k+1}/{args.folds} - Test Acc: {ts_metrics[0]*100:.2f}% | F1: {ts_metrics[1]*100:.2f}% | Rules: {ts_metrics[5]}")

        return results, archive

    # 2. Simple Train-Test Split Mode
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=args.random_state)
        
        train_data = pd.concat([X_train, y_train], axis=1).astype(str)
        test_data = pd.concat([X_test, y_test], axis=1).astype(str)

        moea = MOEA_D_AM_3(
            task='single',
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
            pruning=args.pruning,
            decomposition=args.decomposition,
            archive_type=args.archive_type,
            prediction_strat=args.prediction_strat,
            rulesets=args.rulesets,
            drop_covered=args.drop_covered,
            objs=objs,
            random_state=args.random_state,
            ants_per_subproblem=args.ants
        )

        start_time = time.time()
        moea.run(data=train_data, labels=labels, Val_data=test_data)
        end_time = time.time()
        run_time = end_time - start_time

        priors = moea.get_priors()
        run_archive = moea.get_archive()

        # Save Artifacts
        archive[f"run_{run_id}"] = {
            'archive': run_archive,
            'all_points': moea.get_all_points(),
            'hv_history': moea.get_hv_history(),
            'priors': priors
        }

        # Train metrics
        y_pred_train, _, _ = moea.predict(X=train_data, archive=run_archive, archive_type=args.archive_type, 
                                  prediction_strat=args.prediction_strat, labels=labels, priors=priors)
        tr_metrics = moea.evaluate_slc(y_true=y_train.astype(str), y_pred=y_pred_train)

        results.loc[len(results)] = [
            run_id, 1, 'train', 
            tr_metrics[0]*100, tr_metrics[1]*100, tr_metrics[2]*100, tr_metrics[3]*100, tr_metrics[4]*100,
            tr_metrics[5], tr_metrics[6], tr_metrics[7], run_time
        ]

        # Test metrics
        y_pred_test, _, _ = moea.predict(X=test_data, archive=run_archive, archive_type=args.archive_type, 
                                 prediction_strat=args.prediction_strat, labels=labels, priors=priors)
        ts_metrics = moea.evaluate_slc(y_true=y_test.astype(str), y_pred=y_pred_test)

        results.loc[len(results)] = [
            run_id, 1, 'test', 
            ts_metrics[0]*100, ts_metrics[1]*100, ts_metrics[2]*100, ts_metrics[3]*100, ts_metrics[4]*100,
            ts_metrics[5], ts_metrics[6], ts_metrics[7], run_time
        ]

        print(f"Run {run_id} - Acc: {ts_metrics[0]*100:.2f}% | F1: {ts_metrics[1]*100:.2f}% | Recall: {ts_metrics[2]*100:.2f}% | Precision: {ts_metrics[3]*100:.2f}% |  | Rules: {ts_metrics[5]}")
        
        return results, archive

def main(args: Args) -> None:
    # Directories
    DATA_DIR = dotenv.get_key(env, 'SLC_DATA_DIR')
    RESULTS_DIR = dotenv.get_key(env, 'SLC_RESULTS_DIR')
    MODELS_DIR = dotenv.get_key(env, 'SLC_MODELS_DIR')

    # Load Data (Force string type for rule mining safety)
    try:
        dataframe = pd.read_csv(f"{DATA_DIR}/{args.dataset}.csv", dtype=str)
    except FileNotFoundError:
        print(f"Error: Dataset {args.dataset}.csv not found in {DATA_DIR}")
        return

    # Assuming 'class' is the target column for single label
    labels = ['class']
    if 'class' not in dataframe.columns:
        print("Error: Dataset must contain a 'class' column.")
        return

    X = dataframe.drop(columns=labels)
    y = dataframe['class']

    all_results = pd.DataFrame()
    archive = {}

    print(f"Starting {args.runs} run(s) on {args.dataset}...")
    print(f"Configuration: {args.ants} ants per subproblem, {args.population} subproblems.")

    for run_id in range(1, args.runs+1):
        print(f'\n--- Run {run_id} ---')
        run_results, archive = run_once(args, X, y, labels, run_id, archive, args.objs)
        all_results = pd.concat([all_results, run_results], ignore_index=True)

    # Output Summary
    print(f"\n=== Average Results over {args.runs} runs ===")
    summary = all_results.groupby('split').mean(numeric_only=True)
    print(summary[['accuracy', 'f1_score', 'recall', 'precision', 'specificity', 'nb_rules', 'time']])

    # Save Files
    objs_str = '_'.join([obj[:4] for obj in args.objs])
    
    os.makedirs(f"{RESULTS_DIR}/MOEAAM2/", exist_ok=True)
    os.makedirs(f"{MODELS_DIR}/MOEAAM2/", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/MOEAAM2_IRS/", exist_ok=True)
    os.makedirs(f"{MODELS_DIR}/MOEAAM2_IRS/", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/MOEAAM2_SRS/", exist_ok=True)
    os.makedirs(f"{MODELS_DIR}/MOEAAM2_SRS/", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/MOEAAM2_DC/", exist_ok=True)
    os.makedirs(f"{MODELS_DIR}/MOEAAM2_DC/", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/MOEAAM2_IRS_DC/", exist_ok=True)
    os.makedirs(f"{MODELS_DIR}/MOEAAM2_IRS_DC/", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/MOEAAM2_SRS_DC/", exist_ok=True)
    os.makedirs(f"{MODELS_DIR}/MOEAAM2_SRS_DC/", exist_ok=True)

    if args.archive_type == 'rules':
        if args.drop_covered:
            csv_path = f"{RESULTS_DIR}/MOEAAM2_DC/{args.dataset}_{objs_str}.csv"
            json_path = f"{MODELS_DIR}/MOEAAM2_DC/{args.dataset}_{objs_str}.json"
        else:
            csv_path = f"{RESULTS_DIR}/MOEAAM2/{args.dataset}_{objs_str}.csv"
            json_path = f"{MODELS_DIR}/MOEAAM2/{args.dataset}_{objs_str}.json"
    else:
        if args.rulesets == 'iteration':
            if args.drop_covered:
                csv_path = f"{RESULTS_DIR}/MOEAAM2_IRS_DC/{args.dataset}_{objs_str}.csv"
                json_path = f"{MODELS_DIR}/MOEAAM2_IRS_DC/{args.dataset}_{objs_str}.json"
            else:
                csv_path = f"{RESULTS_DIR}/MOEAAM2_IRS/{args.dataset}_{objs_str}.csv"
                json_path = f"{MODELS_DIR}/MOEAAM2_IRS/{args.dataset}_{objs_str}.json"
        else:
            if args.drop_covered:
                csv_path = f"{RESULTS_DIR}/MOEAAM2_SRS_DC/{args.dataset}_{objs_str}.csv"
                json_path = f"{MODELS_DIR}/MOEAAM2_SRS_DC/{args.dataset}_{objs_str}.json"
            else:
                csv_path = f"{RESULTS_DIR}/MOEAAM2_SRS/{args.dataset}_{objs_str}.csv"
                json_path = f"{MODELS_DIR}/MOEAAM2_SRS/{args.dataset}_{objs_str}.json"

    all_results.to_csv(csv_path, index=False)
    
    # Helper to convert numpy types to native python types for JSON serialization
    def convert(o):
        if isinstance(o, np.int64): return int(o)
        if isinstance(o, np.float64): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)

    with open(json_path, 'w') as f:
        json.dump(archive, f, indent=4, default=convert)
    
    print(f"\nResults saved to: {csv_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="MOEA/D-AM Single Label Trainer")
    
    # Algorithm Parameters
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (without .csv)")
    parser.add_argument("--population", type=int, default=6, help="Number of subproblems (Decomposition size)")
    parser.add_argument("--ants", type=int, default=50, help="Number of ants competing per subproblem")
    parser.add_argument("--neighbors", type=int, default=3, help="T - Neighborhood size")
    parser.add_argument("--groups", type=int, default=2, help="Number of weight groups")
    
    parser.add_argument("--min-examples", type=int, default=10, help="Min coverage per rule")
    parser.add_argument("--max-uncovered", type=int, default=10, help="Max uncovered examples to stop")
    parser.add_argument("--max-iter", type=int, default=100, help="Max iterations")
    
    parser.add_argument("--gamma", type=float, default=0.9, help="Greedy vs Probabilistic selection")
    parser.add_argument("--alpha", type=int, default=1, help="Pheromone weight")
    parser.add_argument("--beta", type=int, default=1, help="Heuristic weight")
    parser.add_argument("--p", type=float, default=0.9, help="Evaporation rate")
    
    parser.add_argument("--pruning", type=int, default=0, help="1=Enable pruning, 0=Disable")
    parser.add_argument("--decomposition", type=str, default="weighted", choices=["weighted", "tchebycheff"])
    parser.add_argument("--archive-type", type=str, default="rules", choices=["rules", "rulesets"])
    parser.add_argument("--rulesets", type=str, default=None, choices=[None, 'iteration', 'subproblem'])
    parser.add_argument("--prediction-strat", type=str, default="all", choices=["all", "best", "voting"])
    parser.add_argument("--drop-covered", type=int, default=0, help="1=Drop covered examples, 0=Do not drop")
    
    # Experiment Parameters
    parser.add_argument("--cross-val", type=int, default=1, help="1=CV, 0=Train/Test")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent runs")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--objs", nargs='+', type=str, default=['specificity', 'sensitivity'], help="Objectives list")

    args = parser.parse_args()
    main(args)
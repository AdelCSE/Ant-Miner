import pandas as pd
import dotenv
import time
import sys
import os
import json

from argparse import ArgumentParser
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from skmultilearn.model_selection import IterativeStratification

sys.path.append("../..")
from src.algorithms.multi_objective.MOEA_D_AM import MOEA_D_AM

env = dotenv.find_dotenv()

@dataclass
class Args:
    task : str
    population : int
    neighbors : int
    groups : int
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
    rulesets_size : int
    prediction_strat : str

    cross_val : bool
    folds : int
    random_state : int
    runs : int


def run_once(args: Args, X, y, labels, run_id: int, archive: dict):
    """
    Executes one run of MOEA/D-AM with or without cross-validation.
    Returns the results dataframe for this run.
    """
    if args.task == 'single':
        results = pd.DataFrame(columns=[
            'run', 'fold', 'accuracy', 'f1_score', 'recall', 'precision',
            'hypervolume', 'nb_rules', 'term_rule_ratio', 'time'
        ])
    else:
        results = pd.DataFrame(columns=[
            'run', 'fold', 'acc', 'f1_score', 'f1_macro', 'f1_micro', 'recall', 'precision', 'hamming_loss',
            'subset_acc', 'ranking_loss', 'coverage', 'avg_precision', 'hypervolume', 
            'nb_rulesets', 'term_rule_ratio', 'time'
        ])
    

    if args.cross_val:

        if args.task == 'single':
            sets = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
        elif args.task == 'multi':
            sets = IterativeStratification(n_splits=args.folds, order=1)
        else:
            raise ValueError("Invalid task type. Choose 'single' or 'multi'!")

        for k, (train_index, test_index) in enumerate(sets.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            moea_d_aco = MOEA_D_AM(
                task=args.task,
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
                rulesets=args.rulesets,
                random_state=args.random_state
            )

            train = pd.concat([X_train, y_train], axis=1)
            test = pd.concat([X_test, y_test], axis=1)

            start_time = time.time()
            moea_d_aco.run(data=train, labels=labels)
            end_time = time.time()

            hv_history = moea_d_aco.get_hv_history()
            all_points = moea_d_aco.get_all_points()
            priors = moea_d_aco.get_priors()

            fold_archive = moea_d_aco.ARCHIVE
            archive[f"run_{run_id}"] = archive.get(f"run_{run_id}", {})
            archive[f"run_{run_id}"][f"fold_{k+1}"] = {
                'archive': fold_archive,
                'all': all_points,
                'hv': hv_history,
                'priors': priors
            }

            if args.task == 'single':
                y_pred, _, _ = moea_d_aco.predict(X=test, archive=fold_archive, archive_type=args.archive_type, prediction_strat=args.prediction_strat, labels=labels, priors=priors, task=args.task)
                acc, f1, recall, precision, nb_rules, term_rule_ratio, hypervolume = moea_d_aco.evaluate_slc(y_true=y_test, y_pred=y_pred)

                results.loc[len(results)] = [
                    run_id, (k % args.folds) + 1, acc*100, f1*100, recall*100, precision*100,
                    hypervolume, nb_rules, term_rule_ratio, end_time - start_time
                ]

                print (f"  Fold {k+1}/{args.folds}: [Acc: {acc*100:.2f}, F1: {f1*100:.2f}, Recall: {recall*100:.2f}, Precision: {precision*100:.2f}, HV: {hypervolume:.2f}, Rules: {nb_rules}, TRR: {term_rule_ratio:.2f}]")
            else:
                y_pred, scores, _ = moea_d_aco.predict(X=test, archive=fold_archive, archive_type=args.archive_type, prediction_strat=args.prediction_strat, labels=labels, priors=priors, task=args.task)
                metrics = moea_d_aco.evaluate_mlc(y_true=y_test.to_numpy(), y_pred=y_pred.to_numpy(), labels=labels, scores=scores.to_numpy())
                results.loc[len(results)] = [
                    run_id, (k % args.folds) + 1, metrics['acc']*100, metrics['f1_score']*100, metrics['f1_macro']*100, 
                    metrics['f1_micro']*100, metrics['recall']*100, metrics['precision']*100, metrics['hamming_loss'],
                    metrics['subset_acc']*100, metrics['ranking_loss'], metrics['coverage'], 
                    metrics['avg_precision']*100, metrics['hypervolume'], metrics['nb_rulesets'], 
                    metrics['term_rule_ratio'], end_time - start_time
                ]

                print(f'Fold {k+1}/{args.folds}: [Acc: {metrics['acc']*100:.2f}, F1 Score: {metrics['f1_score']*100:.2f}, F1 Macro: {metrics['f1_macro']*100:.2f}, F1 Micro: {metrics['f1_micro']*100:.4f}, Recall: {metrics['recall']*100:.4f}, Precision: {metrics['precision']*100:.4f}, Hamming Loss: {metrics['hamming_loss']:.4f}, Subset Acc: {metrics['subset_acc']*100:.2f}, Ranking Loss: {metrics['ranking_loss']:.4f}, Coverage: {metrics['coverage']:.4f}, Avg Precision: {metrics['avg_precision']*100:.2f}, HV: {metrics['hypervolume']:.2f}, Rulesets: {metrics['nb_rulesets']}, TRR: {metrics['term_rule_ratio']:.2f}]')
        return results, archive
    else:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=args.random_state
        )

        moea_d_aco = MOEA_D_AM(
            task = args.task,
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
            rulesets=args.rulesets,
            random_state=args.random_state
        )
        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)
        # to string
        train = train.astype(str)
        test = test.astype(str)
        start_time = time.time()
        moea_d_aco.run(data=train, labels=labels)
        end_time = time.time()
        time_taken = end_time - start_time
        run_archive = moea_d_aco.ARCHIVE
        hv_history = moea_d_aco.get_hv_history()
        all_points = moea_d_aco.get_all_points()
        priors = moea_d_aco.get_priors()

        archive[f"run_{run_id}"] = {
            'archive': run_archive,
            'all': all_points,
            'hv': hv_history,
            'priors': priors
        }
        if args.task == 'single':
            y_pred, _, _ = moea_d_aco.predict(X=test, archive=run_archive, archive_type=args.archive_type, prediction_strat=args.prediction_strat, labels=labels, priors=priors, task=args.task)
            acc, f1, recall, precision, nb_rules, term_rule_ratio, hypervolume = moea_d_aco.evaluate_slc(y_true=y_test, y_pred=y_pred)

            results.loc[len(results)] = [
                run_id, 1, acc*100, f1*100, recall*100, precision*100,
                hypervolume, nb_rules, term_rule_ratio, time_taken
            ]

            print (f"  [Acc: {acc*100:.2f}, F1: {f1*100:.2f}, Recall: {recall*100:.2f}, Precision: {precision*100:.2f}, HV: {hypervolume:.2f}, Rules: {nb_rules}, TRR: {term_rule_ratio:.2f}]")
        else:
            y_pred, scores, _ = moea_d_aco.predict(X=test, archive=run_archive, archive_type=args.archive_type, prediction_strat=args.prediction_strat, labels=labels, priors=priors, task=args.task)
            metrics = moea_d_aco.evaluate_mlc(y_true=y_test.to_numpy(), y_pred=y_pred.to_numpy(), labels=labels, scores=scores.to_numpy())
            results.loc[len(results)] = [
                run_id, 1, metrics['acc']*100, metrics['f1_score']*100, metrics['f1_macro']*100, 
                metrics['f1_micro']*100, metrics['recall']*100, metrics['precision']*100, metrics['hamming_loss'],
                metrics['subset_acc']*100, metrics['ranking_loss'], metrics['coverage'], 
                metrics['avg_precision']*100, metrics['hypervolume'], metrics['nb_rulesets'], 
                metrics['term_rule_ratio'], time_taken
            ]

            print(f'[Acc: {metrics["acc"]*100:.2f}, F1 Score: {metrics["f1_score"]*100:.2f}, F1 Macro: {metrics["f1_macro"]*100:.2f}, F1 Micro: {metrics["f1_micro"]*100:.2f}, Recall: {metrics["recall"]*100:.2f}, Precision: {metrics["precision"]*100:.2f}, Hamming Loss: {metrics["hamming_loss"]:.4f}, Subset Acc: {metrics["subset_acc"]*100:.2f}, Ranking Loss: {metrics["ranking_loss"]:.4f}, Coverage: {metrics["coverage"]:.4f}, Avg Precision: {metrics["avg_precision"]*100:.2f}, HV: {metrics["hypervolume"]:.2f}, Rulesets: {metrics["nb_rulesets"]}, TRR: {metrics["term_rule_ratio"]:.2f}]')

        return results, archive
    
def main(args: Args) -> None:

    # Get env variables
    DATA_DIR = dotenv.get_key(env, 'SLC_DATA_DIR') if args.task == 'single' else dotenv.get_key(env, 'MLC_DATA_DIR')
    RESULTS_DIR = dotenv.get_key(env, 'SLC_RESULTS_DIR') if args.task == 'single' else dotenv.get_key(env, 'MLC_RESULTS_DIR')
    ARCHIVE_DIR = dotenv.get_key(env, 'SLC_ARCHIVE_DIR') if args.task == 'single' else dotenv.get_key(env, 'MLC_ARCHIVE_DIR')

    # Load dataset
    if args.task == 'single':
        dataframe = pd.read_csv(DATA_DIR, dtype=str)
    else:
        dataframe = pd.read_csv(DATA_DIR)

    labels = ['class'] if args.task == 'single' else [name for name in dataframe.columns if 'label' in name]
    X = dataframe.drop(columns=labels)
    y = dataframe[labels] if args.task == 'multi' else dataframe['class']

    dataset = DATA_DIR.split('/')[-1].split('.')[0]
    pruning_sfx = '_p' if args.pruning else '_np'
    decomposition_sfx =  '_ws' if args.decomposition == 'weighted' else '_tch'

    all_results = pd.DataFrame()
    archive = {}

    for run_id in range(1, args.runs+1):
        print(f"\n=== Starting Run {run_id}/{args.runs} ===\n")
        run_results, archive = run_once(args, X, y, labels, run_id, archive)
        all_results = pd.concat([all_results, run_results], ignore_index=True)

        if args.task == 'single':
            print (f"Run {run_id} completed in {run_results['time'].mean():.2f} seconds: [Acc: {run_results['accuracy'].mean():.2f} ± {run_results['accuracy'].std():.2f}, F1: {run_results['f1_score'].mean():.2f} ± {run_results['f1_score'].std():.2f}, Recall: {run_results['recall'].mean():.2f} ± {run_results['recall'].std():.2f}, Precision: {run_results['precision'].mean():.2f} ± {run_results['precision'].std():.2f}, HV: {run_results['hypervolume'].mean():.2f} ± {run_results['hypervolume'].std():.2f}, Rules: {run_results['nb_rules'].mean():.2f} ± {run_results['nb_rules'].std():.2f}, TRR: {run_results['term_rule_ratio'].mean():.2f} ± {run_results['term_rule_ratio'].std():.2f}]")
        else:
            print (f"Run {run_id} completed in {run_results['time'].mean():.2f} seconds: [Acc: {run_results['acc'].mean():.2f} ± {run_results['acc'].std():.2f}, F1 Score: {run_results['f1_score'].mean():.2f} ± {run_results['f1_score'].std():.2f}, F1 Macro: {run_results['f1_macro'].mean():.2f} ± {run_results['f1_macro'].std():.2f}, F1 Micro: {run_results['f1_micro'].mean():.2f} ± {run_results['f1_micro'].std():.2f}, Recall: {run_results['recall'].mean():.2f} ± {run_results['recall'].std():.2f}, Precision: {run_results['precision'].mean():.2f} ± {run_results['precision'].std():.2f}, Hamming Loss: {run_results['hamming_loss'].mean():.4f} ± {run_results['hamming_loss'].std():.4f}, Subset Acc: {run_results['subset_acc'].mean():.2f} ± {run_results['subset_acc'].std():.2f}, Ranking Loss: {run_results['ranking_loss'].mean():.4f} ± {run_results['ranking_loss'].std():.4f}, Coverage: {run_results['coverage'].mean():.4f} ± {run_results['coverage'].std():.4f}, Avg Precision: {run_results['avg_precision'].mean():.2f} ± {run_results['avg_precision'].std():.2f}, HV: {run_results['hypervolume'].mean():.2f} ± {run_results['hypervolume'].std():.2f}, Rulesets: {run_results['nb_rulesets'].mean():.2f} ± {run_results['nb_rulesets'].std():.2f}, TRR: {run_results['term_rule_ratio'].mean():.2f} ± {run_results['term_rule_ratio'].std():.2f}]")


    folder = "CV" if args.cross_val else "ALL"
    # Save archive
    archive_path = f"{ARCHIVE_DIR}/MOEA_D_AM/{folder}"
    os.makedirs(archive_path, exist_ok=True)

    # Save aggregated results
    save_path = f"{RESULTS_DIR}/MOEA_D_AM/{folder}"
    os.makedirs(save_path, exist_ok=True)

    # print average results
    if args.task == 'single':
        print(f"\nAverage results over {args.runs} runs:")
        print(f" - Avg. Time: {all_results['time'].mean():.2f} ± {all_results['time'].std():.2f}")
        print(f" - Avg. Accuracy: {all_results['accuracy'].mean():.2f} ± {all_results['accuracy'].std():.2f}")
        print(f" - Avg. F1 Score: {all_results['f1_score'].mean():.2f} ± {all_results['f1_score'].std():.2f}")
        print(f" - Avg. Recall: {all_results['recall'].mean():.2f} ± {all_results['recall'].std():.2f}")
        print(f" - Avg. Precision: {all_results['precision'].mean():.2f} ± {all_results['precision'].std():.2f}")
        print(f" - Avg. Hypervolume: {all_results['hypervolume'].mean():.2f} ± {all_results['hypervolume'].std():.2f}")
        print(f" - Avg. Rules: {all_results['nb_rules'].mean():.2f} ± {all_results['nb_rules'].std():.2f}")
        print(f" - Avg. Term-Rule Ratio: {all_results['term_rule_ratio'].mean():.2f} ± {all_results['term_rule_ratio'].std():.2f}")
    else:
        print(f"\nAverage results over {args.runs} runs:")
        print(f" - Avg. Time: {all_results['time'].mean():.2f} ± {all_results['time'].std():.2f}")
        print(f" - Avg. Accuracy: {all_results['acc'].mean():.2f} ± {all_results['acc'].std():.2f}")
        print(f" - Avg. F1 Score: {all_results['f1_score'].mean():.2f} ± {all_results['f1_score'].std():.2f}")
        print(f" - Avg. F1 Macro: {all_results['f1_macro'].mean():.2f} ± {all_results['f1_macro'].std():.2f}")
        print(f" - Avg. F1 Micro: {all_results['f1_micro'].mean():.2f} ± {all_results['f1_micro'].std():.2f}")
        print(f" - Avg. Recall: {all_results['recall'].mean():.2f} ± {all_results['recall'].std():.2f}")
        print(f" - Avg. Precision: {all_results['precision'].mean():.2f} ± {all_results['precision'].std():.2f}")
        print(f" - Avg. Hamming Loss: {all_results['hamming_loss'].mean():.4f} ± {all_results['hamming_loss'].std():.4f}")
        print(f" - Avg. Subset Accuracy: {all_results['subset_acc'].mean():.2f} ± {all_results['subset_acc'].std():.2f}")
        print(f" - Avg. Ranking Loss: {all_results['ranking_loss'].mean():.4f} ± {all_results['ranking_loss'].std():.4f}")
        print(f" - Avg. Coverage: {all_results['coverage'].mean():.4f} ± {all_results['coverage'].std():.4f}")
        print(f" - Avg. Average Precision: {all_results['avg_precision'].mean():.2f} ± {all_results['avg_precision'].std():.2f}")
        print(f" - Avg. Hypervolume: {all_results['hypervolume'].mean():.2f} ± {all_results['hypervolume'].std():.2f}")
        print(f" - Avg. Rulesets: {all_results['nb_rulesets'].mean():.2f} ± {all_results['nb_rulesets'].std():.2f}")
        print(f" - Avg. Term-Rule Ratio: {all_results['term_rule_ratio'].mean():.2f} ± {all_results['term_rule_ratio'].std():.2f}")


    
    with open(f"{archive_path}/{dataset}{decomposition_sfx}{pruning_sfx}_r{args.runs}.json", 'w') as f:
        json.dump(archive, f, indent=4)
    
    all_results.to_csv(f"{save_path}/{dataset}{decomposition_sfx}{pruning_sfx}_r{args.runs}.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="AntMiner Algorithm with Multiple Runs")
    
    # extend to multi-label
    parser.add_argument("--task", type=str, default="single", choices=["single", "multi"])
    parser.add_argument("--population", type=int, default=50, help="Population size")
    parser.add_argument("--neighbors", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--groups", type=int, default=5, help="Number of groups")
    parser.add_argument("--min-examples", type=int, default=10, help="Minimum examples per rule")
    parser.add_argument("--max-uncovered", type=int, default=10, help="Number of examples left uncovered to stop training")
    parser.add_argument("--max-iter", type=int, default=100, help="Number of iterations")
    parser.add_argument("--gamma", type=float, default=0.9, help="Exploration/Exploitation control parameter")
    parser.add_argument("--alpha", type=int, default=1, help="Pheromone influence parameter")
    parser.add_argument("--beta", type=int, default=1, help="Heuristic influence parameter")
    parser.add_argument("--p", type=float, default=0.9, help="Pheromone evaporation rate")
    parser.add_argument("--pruning", type=int, default=0, help="Use rule pruning (1) or not (0)")

    parser.add_argument("--decomposition", type=str, default="weighted", choices=["weighted", "tchebycheff"], help="Decomposition method")
    parser.add_argument("--archive-type", type=str, default="rules", choices=["rules", "rulesets"], help="Structure of the archive")
    parser.add_argument("--rulesets", type=str, default=None, choices=[None, 'iteration', 'subproblem'], help="Ruleset formation strategy")
    parser.add_argument("--ruleset-size", type=int, default=2, help="Number of rules per ruleset (if rulesets formation is used)")
    parser.add_argument("--prediction-strat", type=str, default="all", choices=["all", "best", "reference", "voting"], help="Prediction strategy")

    parser.add_argument("--cross-val", type=int, default=0, help="Use cross-validation (1) or train-test split (0)")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--random-state", type=int, default=None, help="Random state for reproducibility")

    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs")

    args = parser.parse_args()
    main(args)

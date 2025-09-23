import sys
import pandas as pd
import optuna
import dotenv
from sklearn.model_selection import StratifiedKFold

sys.path.append("../..")
from src.algorithms.multi_objective.MOEA_D_AM import MOEA_D_AM

env = dotenv.find_dotenv()

def objective(trial):
    population = trial.suggest_int('population_size', 50, 200)
    neighbors = trial.suggest_int('neighborhood_size', 2, 10)
    groups = trial.suggest_int('groups', 2, 5)
    min_examples = 10
    max_uncovered = 10
    max_iter = trial.suggest_int('max_iter', 100, 500)
    gamma = trial.suggest_float('gamma', 0.5, 0.9)
    delta = 0.5
    alpha = trial.suggest_int('alpha', 1, 5)
    beta = trial.suggest_int('beta', 1, 5)
    p = trial.suggest_float('p', 0.5, 0.9)
    pruning = 0
    archive_type = 'rulesets'
    rulesets = 'iteration'
    random_state = 42

    # Load data
    X, y = load_data()

    moea_d_aco = MOEA_D_AM(
                population=population,
                neighbors=neighbors,
                groups=groups,
                min_examples=min_examples,
                max_uncovered=max_uncovered,
                max_iter=max_iter,
                p=p,
                gamma=gamma,
                alpha=alpha,
                beta=beta,
                delta=delta,
                pruning=pruning,
                archive_type=archive_type,
                rulesets=rulesets,
                random_state=random_state
    )

    # 10-fold cross-validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    f1_scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        moea_d_aco.run(X_train, y_train)
        _, f1, _, _, _ = moea_d_aco.evaluate(X=X_test, y=y_test, prediction_strat='best')
        f1_scores.append(f1)

    avg_f1_score = sum(f1_scores) / len(f1_scores)
    return avg_f1_score

def load_data():
    DATA_DIR = dotenv.get_key(env, 'DATA_DIR')

    dataframe = pd.read_csv(DATA_DIR)

    X = dataframe.drop('class', axis=1)
    y = dataframe['class'].astype(str)

    return X, y


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
import optuna

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score


SEED = 0

def objective_lr(trial, X_train, y_train, X_valid, y_valid, column_transformer, multi):
    if multi:
        possible_penalties = ["l2"]
    else:
        possible_penalties = ["l1", "l2"]
    params = {
        "penalty": trial.suggest_categorical("penalty", possible_penalties),
        "C": trial.suggest_categorical("C", [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 1e-05])
    }

    if multi:
        solver = "lbfgs"
        multi_class = "multinomial"
    else:
        solver = "liblinear" if params["penalty"] == "l1" else "lbfgs"
        multi_class = "ovr"

    pipeline = Pipeline([
        ("ohe_and_scaling", column_transformer),
        ("classifier", LogisticRegression(**params, solver=solver, max_iter=1000, random_state=SEED, multi_class=multi_class))
    ])
    model = pipeline.fit(X_train, y_train)

    if multi:
        y_pred = model.predict_proba(X_valid)
        score = roc_auc_score(y_valid, y_pred, multi_class="ovr", average="macro")
    else:
        y_pred = model.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, y_pred)
    return score


def run_lr_tuning(X_train, y_train, X_valid, y_valid, column_transformer, n_trials=20, multi=False):
    study = optuna.create_study(study_name="logistic_regression_optimization", direction="maximize")
    study.optimize(lambda trial: objective_lr(trial, X_train, y_train, X_valid, y_valid, column_transformer, multi), n_trials=n_trials)
    best_params = study.best_trial.params
    solver = "liblinear" if best_params["penalty"] == "l1" else "lbfgs"
    multi_class = "multinomial" if multi else "ovr"
    pipeline = Pipeline([
        ("ohe_and_scaling", column_transformer),
        ("classifier", LogisticRegression(**best_params, solver=solver, max_iter=1000, random_state=SEED, multi_class=multi_class))
    ])
    model = pipeline.fit(X_train, y_train)
    return model


def objective_catboost(trial, X_train, y_train, X_valid, y_valid, column_transformer, multi):
    params = {
        "depth": trial.suggest_categorical("depth", [2, 4, 6, 8, 10]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.03, 0.1, 0.3]),
        "l2_leaf_reg": trial.suggest_categorical("l2_leaf_reg", [0.001, 0.01, 0.1, 1.0, 3.0, 6.0, 10.0])
    }
    loss_function = "MultiClass" if multi else "Logloss"
    pipeline = Pipeline([
        ("ohe_and_scaling", column_transformer),
        ("classifier", CatBoostClassifier(**params, loss_function=loss_function, iterations=1000, random_state=SEED, verbose=0))
    ])
    model = pipeline.fit(X_train, y_train)

    if multi:
        y_pred = model.predict_proba(X_valid)
        score = roc_auc_score(y_valid, y_pred, multi_class="ovr", average="macro")
    else:
        y_pred = model.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, y_pred)
    return score

def run_catboost_tuning(X_train, y_train, X_valid, y_valid, column_transformer, n_trials=20, multi=False):
    study = optuna.create_study(study_name="catboost_optimization", direction="maximize")
    study.optimize(lambda trial: objective_catboost(trial, X_train, y_train, X_valid, y_valid, column_transformer, multi), n_trials=n_trials)
    loss_function = "MultiClass" if multi else "Logloss"
    pipeline = Pipeline([
        ("ohe_and_scaling", column_transformer),
        ("clf", CatBoostClassifier(**study.best_trial.params, loss_function=loss_function, iterations=1000, random_state=SEED, verbose=0))
    ])
    model = pipeline.fit(X_train, y_train)
    return model


def objective_lgbm(trial, X_train, y_train, X_valid, y_valid, column_transformer, multi):
    params = {
        "num_leaves": trial.suggest_categorical("num_leaves", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]),
        "reg_alpha": trial.suggest_categorical("reg_alpha", [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.03, 0.1, 0.3])
    }
    pipeline = Pipeline([
        ("ohe_and_scaling", column_transformer),
        ("classifier", LGBMClassifier(**params, n_estimators=1000, random_state=SEED, verbose=-1))
    ])
    model = pipeline.fit(X_train, y_train)

    if multi:
        y_pred = model.predict_proba(X_valid)
        score = roc_auc_score(y_valid, y_pred, multi_class="ovr", average="macro")
    else:
        y_pred = model.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, y_pred)
    return score

def run_lgbm_tuning(X_train, y_train, X_valid, y_valid, column_transformer, n_trials=20, multi=False):
    study = optuna.create_study(study_name="lgdm_optimization", direction="maximize")
    study.optimize(lambda trial: objective_lgbm(trial, X_train, y_train, X_valid, y_valid, column_transformer, multi), n_trials=n_trials)
    pipeline = Pipeline([
        ("ohe_and_scaling", column_transformer),
        ("classifier", LGBMClassifier(**study.best_trial.params, n_estimators=1000, random_state=SEED, verbose=-1))
    ])
    model = pipeline.fit(X_train, y_train)
    return model

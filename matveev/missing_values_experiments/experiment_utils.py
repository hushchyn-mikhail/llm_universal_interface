import openml
import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, 
    precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


SEED = 0

def load_openml(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
    y = LabelEncoder().fit_transform(y)
    categorical_features = [c for c, is_cat in zip(attribute_names, categorical_indicator) if is_cat]
    return X, y, categorical_features

def objective_catboost(trial, X_train, y_train, X_valid, y_valid, categorical_features, multi):
    params = {
        "depth": trial.suggest_categorical("depth", [2, 4, 6, 8, 10]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.001, 0.01, 0.03, 0.1, 0.3]),
        "l2_leaf_reg": trial.suggest_categorical("l2_leaf_reg", [0.001, 0.01, 0.1, 1.0, 3.0, 6.0])
    }
    loss_function = "MultiClass" if multi else "Logloss"
    model = CatBoostClassifier(**params, 
                               iterations=1000, 
                               random_state=SEED, 
                               verbose=0, 
                               allow_writing_files=False, 
                               cat_features=categorical_features, 
                               loss_function=loss_function, 
                               nan_mode="Min")
    model.fit(X_train, y_train)
    if multi:
        score = roc_auc_score(y_valid, model.predict_proba(X_valid), multi_class="ovr", average="macro")
    else:
        score = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
    return score

def bootstrap_metrics(y_true, y_pred, y_prob, n_iter, multi):
    scores = []
    for i in range(n_iter):
        y_true_boot, y_pred_boot, y_prob_boot = resample(y_true, y_pred, y_prob, random_state=i+1)
        try:
            if multi:
                auc = roc_auc_score(y_true_boot, y_prob_boot, multi_class="ovr", average="macro")
                f1 = f1_score(y_true_boot, y_pred_boot, average="macro")
                pr = precision_score(y_true_boot, y_pred_boot, average="macro", zero_division=0)
                rc = recall_score (y_true_boot, y_pred_boot, average="macro", zero_division=0)
            else:
                auc = roc_auc_score(y_true_boot, y_prob_boot)
                f1  = f1_score(y_true_boot, y_pred_boot)
                pr  = precision_score(y_true_boot, y_pred_boot, zero_division=0)
                rc  = recall_score (y_true_boot, y_pred_boot, zero_division=0)
            scores.append((auc, f1, accuracy_score(y_true_boot, y_pred_boot), pr, rc))
        except ValueError:
            # пропускаем сэмпл, где не представлены все классы
            continue
    scores = np.asarray(scores)
    means, stds = scores.mean(0), scores.std(0, ddof=1)
    names = ["ROC-AUC", "F1", "Accuracy", "Precision", "Recall"]
    return {n: f"{m:.4f}±{s:.4f}" for n, m, s in zip(names, means, stds)}

def run_experiment(dataset_id=0, missing_rates=(0.2,0.5,0.9), n_trials=20, n_boot=1000, multi=False, X=None, y=None, categorical_features=None):
    if X is None:
        X, y, categorical_features = load_openml(dataset_id)
    res = []
    for missing_rate in missing_rates:
        X_miss = X.mask(np.random.rand(*X.shape) < missing_rate)
        X_miss[categorical_features] = X_miss[categorical_features].astype(str)
        
        X_train, X_test, y_train, y_test = train_test_split(X_miss, y, test_size=0.2, random_state=SEED, stratify=y, shuffle=True)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=SEED, stratify=y_test, shuffle=True)
        
        study = optuna.create_study(study_name=f"catboost_optimization_{missing_rate}", direction="maximize")
        study.optimize(lambda trial: objective_catboost(trial, X_train, y_train, X_valid, y_valid, categorical_features, multi), n_trials=n_trials)
        
        model_cb = CatBoostClassifier(**study.best_trial.params, 
                                      iterations=1000, 
                                      random_state=SEED, 
                                      verbose=0, 
                                      allow_writing_files=False, 
                                      cat_features=categorical_features, 
                                      loss_function="MultiClass" if multi else "Logloss",
                                      nan_mode="Min")
        model_cb.fit(X_train, y_train)

        y_pred = model_cb.predict(X_test)
        y_prob = model_cb.predict_proba(X_test)
        if not multi:
            y_prob = y_prob[:,1] 
        metrics = bootstrap_metrics(y_test, y_pred, y_prob, n_boot, multi)
        res.append({"missing_rate": missing_rate} | metrics)

    res = pd.DataFrame(res)
    return res
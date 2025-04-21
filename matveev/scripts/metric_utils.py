import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.utils import resample

def bootstrap_metrics(y_true, y_pred, y_prob, n_iter=1000, multi=False):
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

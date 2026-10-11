"""
Re-evaluate the shipped credit-risk artifact on the notebook's held-out test split.

Reproduces the cleaning and the stratified 80/20 split from Credit_Risk.ipynb
(cells 20–23, random_state=42), then scores credit_risk_model.pkl exactly as
main.py does (predict_proba[:, 1] >= best_threshold.pkl). Nothing is retrained.

Writes static/data/evaluation.json (metrics, curves, permutation importance)
and static/data/schema.json (feature ranges/categories for form validation).

Run from the project root:  python evaluation/evaluate_artifact.py
"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

FEATURES = ["person_age", "person_income", "person_home_ownership", "person_emp_length", "loan_intent",
            "loan_grade", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_default_on_file",
            "cb_person_cred_hist_length"]
CATEGORICAL = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]


def notebook_split():
    df = pd.read_csv("credit_risk_dataset.csv")
    raw_rows = len(df)
    d = df.copy()
    d.drop_duplicates(inplace=True)
    d = d[d["person_age"] >= 18]
    d = d[d["person_age"] <= 100]
    d = d[d["person_emp_length"] <= d["person_age"]]
    d = d[d["person_emp_length"] <= 60]
    d = d[d["loan_amnt"] > 0]
    X = d.drop("loan_status", axis=1)
    y = d["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return df, raw_rows, d, X_train, X_test, y_train, y_test


def at_threshold(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(t),
        "accuracy": float((pred == y).mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "flagged_rate": float(pred.mean()),
    }


def thin(xs, ys, n=120):
    idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
    return [[float(xs[i]), float(ys[i])] for i in idx]


def main():
    model = joblib.load("credit_risk_model.pkl")
    threshold = float(joblib.load("best_threshold.pkl"))
    df, raw_rows, clean, X_train, X_test, y_train, y_test = notebook_split()
    y = y_test.to_numpy()
    p = model.predict_proba(X_test[FEATURES])[:, 1]

    prec, rec, _ = precision_recall_curve(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10)
    sweep = [at_threshold(y, p, t) for t in np.round(np.arange(0.05, 0.96, 0.05), 2)]
    best_f1 = max(sweep, key=lambda r: r["f1"])

    prob_bins = []
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        prob_bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                          "observed_default_rate": float(y[m].mean()) if m.sum() else None,
                          "mean_predicted": float(p[m].mean()) if m.sum() else None})

    # Permutation importance on the test split (drop in ROC-AUC), 5 repeats, fixed seed.
    rng = np.random.default_rng(0)
    base_auc = roc_auc_score(y, p)
    importance = []
    for col in FEATURES:
        drops = []
        for _ in range(5):
            Xp = X_test[FEATURES].copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base_auc - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        importance.append({"feature": col, "mean": float(np.mean(drops)), "std": float(np.std(drops))})
    importance.sort(key=lambda r: -r["mean"])

    evaluation = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "evaluation/evaluate_artifact.py",
        "artifact": "credit_risk_model.pkl (CalibratedClassifierCV, sigmoid, cv=5, around the tuned XGBoost pipeline)",
        "data": {
            "raw_rows": raw_rows, "rows_after_cleaning": int(len(clean)),
            "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
            "test_positives": int(y.sum()), "test_negatives": int((1 - y).sum()),
            "default_rate_raw": float(df["loan_status"].mean()),
            "missing_raw": {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().sum()},
        },
        "ranking": {"roc_auc": float(base_auc), "pr_auc": float(average_precision_score(y, p)),
                    "brier": float(brier_score_loss(y, p))},
        "stored_threshold": at_threshold(y, p, threshold),
        "threshold_0_5": at_threshold(y, p, 0.5),
        "best_f1_on_grid": best_f1,
        "threshold_sweep": sweep,
        "curves": {
            "pr": thin(rec, prec), "roc": thin(fpr, tpr),
            "calibration": [[float(a), float(b)] for a, b in zip(mean_pred, frac_pos)],
        },
        "probability_bins": prob_bins,
        "permutation_importance": {"metric": "drop in ROC-AUC", "repeats": 5, "features": importance},
        "notebook_recorded": {
            "source": "Credit_Risk.ipynb saved outputs (cells 39, 43, 52, 78)",
            "cv_5fold_train": {
                "Logistic Regression": {"roc_auc": 0.8705, "accuracy": 0.8116, "precision": 0.5448, "recall": 0.7781, "f1": 0.6408},
                "XGBoost (untuned)": {"roc_auc": 0.9394, "accuracy": 0.9087, "precision": 0.7913, "recall": 0.7857, "f1": 0.7880},
            },
            "test_default_threshold": {
                "Logistic Regression (baseline)": {"accuracy": 0.82, "precision": 0.56, "recall": 0.79, "f1": 0.65},
                "XGBoost (tuned, uncalibrated)": {"accuracy": 0.92, "precision": 0.81, "recall": 0.81, "f1": 0.81},
            },
            "tuned_xgb_errors": {"false_positives": 258, "false_negatives": 256},
            "tuning": "RandomizedSearchCV, 150 candidates × 5 folds, scoring = average precision; best CV score 0.90",
        },
    }

    clean_X = clean[FEATURES]
    schema = {"features": {}}
    for c in FEATURES:
        s = clean_X[c]
        if c in CATEGORICAL:
            schema["features"][c] = {"type": "category", "values": sorted(s.dropna().unique().tolist()),
                                     "counts": {k: int(v) for k, v in s.value_counts().items()}}
        else:
            schema["features"][c] = {"type": "number", "min": float(s.min()), "max": float(s.max()),
                                     "p01": float(s.quantile(0.01)), "p50": float(s.median()), "p99": float(s.quantile(0.99)),
                                     "missing_in_raw": int(df[c].isna().sum())}

    os.makedirs("static/data", exist_ok=True)
    with open("static/data/evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)
    with open("static/data/schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    st = evaluation["stored_threshold"]
    print(f"test n={len(y)} (pos {y.sum()}), ROC-AUC={base_auc:.4f}, PR-AUC={evaluation['ranking']['pr_auc']:.4f}")
    print(f"stored threshold {threshold:.4f}: acc={st['accuracy']:.4f} P={st['precision']:.4f} R={st['recall']:.4f} F1={st['f1']:.4f}")
    print(f"best F1 on grid: t={best_f1['threshold']} F1={best_f1['f1']:.4f}")
    print("top features:", [(r['feature'], round(r['mean'], 4)) for r in importance[:5]])


if __name__ == "__main__":
    main()

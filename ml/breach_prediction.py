# ml/breach_prediction.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance

def _choose_threshold_on_validation(y_val: np.ndarray, proba_val: np.ndarray, target_recall: float = 0.80) -> float:
    thresholds = np.unique(proba_val)[::-1]
    y_val = np.asarray(y_val).astype(int)
    proba_val = np.asarray(proba_val).astype(float)

    if y_val.sum() == 0:
        return 0.50

    # Track best recall in case target recall not achievable
    best_t = 0.50
    best_recall = -1.0

    # Collect candidates that meet target recall
    candidates = []
    for t in thresholds:
        y_hat = (proba_val >= t).astype(int)
        tp = int(((y_hat == 1) & (y_val == 1)).sum())
        fn = int(((y_hat == 0) & (y_val == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if recall > best_recall:
            best_recall = recall
            best_t = float(t)

        if recall >= target_recall:
            alerts = int((y_hat == 1).sum())
            candidates.append((alerts, float(t), recall))

    if candidates:
        candidates.sort(key=lambda x: (x[0], -x[1]))
        return candidates[0][1]

    return best_t


def run_task6_breach_prediction(
    file_path: str,
    seed: int = 42,
    target_recall: float = 0.80
) -> None:
    """
    Task 6:
    Predict Breachornot using Logistic Regression, avoiding leakage by excluding LoS.
    Uses 70–15–15 (train/val/test). Threshold is chosen on VAL to hit a target recall for breaches.
    """

    # 1) Load
    df = pd.read_csv(file_path).copy()

    # 2) Target
    df["Breachornot"] = df["Breachornot"].astype(str).str.strip().str.lower()
    y = (df["Breachornot"] == "breach").astype(int)

    # 3) Features (avoid leakage)
    drop_cols = ["Breachornot", "LoS", "ID"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 4) Column types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # 5) Preprocess
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    # 6) Logistic Regression 
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear"
    )
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # 7) 70–15–15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=seed,
        stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp
    )

    print("\n" + "=" * 90)
    print("TASK 6 — Predicting 4-hour breach (ML, no leakage)")
    print("=" * 90)
    print(f"Train: {len(y_train)} | breach rate: {y_train.mean()*100:.2f}%")
    print(f"Val:   {len(y_val)} | breach rate: {y_val.mean()*100:.2f}%")
    print(f"Test:  {len(y_test)} | breach rate: {y_test.mean()*100:.2f}%")

    # 8) Fit on train
    clf.fit(X_train, y_train)

    # 9) Threshold selection on validation
    val_proba = clf.predict_proba(X_val)[:, 1]
    threshold = _choose_threshold_on_validation(y_val, val_proba, target_recall=target_recall)

    val_pred = (val_proba >= threshold).astype(int)
    print("\n" + "=" * 90)
    print("THRESHOLD SELECTION (VALIDATION SET)")
    print("=" * 90)
    print(f"Target recall: {target_recall:.2f}")
    print(f"Chosen threshold: {threshold:.4f}")
    print("Validation confusion matrix:")
    print(confusion_matrix(y_val, val_pred))
    print("Validation classification report:")
    print(classification_report(y_val, val_pred, digits=3))

    # 10) Evaluate on test
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 90)
    print("TEST PERFORMANCE (final reporting)")
    print("=" * 90)
    print(f"Threshold used: {threshold:.4f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.3f}")
    print("\nConfusion Matrix (TEST):")
    print(cm)
    print("\nClassification report (TEST):")
    print(classification_report(y_test, y_pred, digits=3))

    # 11) Visualisations
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve — Breach Prediction (TEST)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision–Recall Curve — Breach Prediction (TEST)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix (TEST) — threshold={threshold:.3f}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Non-breach", "Breach"])
    plt.yticks([0, 1], ["Non-breach", "Breach"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.show()

    # 12) Key factors: permutation importance
    result = permutation_importance(
        clf, X_test, y_test,
        n_repeats=10,
        random_state=seed,
        scoring="average_precision"
    )

    # NOTE:
    # We compute permutation importance on the *full pipeline* (clf) using the *raw* X_test DataFrame.
    # In this case, sklearn permutes one raw input column at a time, so the importance vector length
    # equals the number of columns in X_test (NOT the one-hot expanded feature space). If we attempt
    # to build one-hot feature names here, we can get a length mismatch and pandas will raise:
    #   ValueError: All arrays must be of the same length
    # Therefore, use the raw column names for the importance table.
    if hasattr(X_test, "columns"):
        feature_names = list(X_test.columns)
    else:
        # Fallback (e.g., if X_test is a numpy array)
        feature_names = list(numeric_features) + list(categorical_features)

    # Defensive trimming in case of any unexpected mismatch
    m = min(len(feature_names), len(result.importances_mean), len(result.importances_std))
    imp_df = pd.DataFrame({
        "Feature": feature_names[:m],
        "ImportanceMean": result.importances_mean[:m],
        "ImportanceStd": result.importances_std[:m],
    }).sort_values("ImportanceMean", ascending=False)

    print("\nTop 15 predictors (Permutation importance, scored by PR-AUC):")
    print(imp_df.head(15).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    top = imp_df.head(15).iloc[::-1]
    plt.figure()
    plt.barh(top["Feature"], top["ImportanceMean"])
    plt.title("Top Predictors of Breach (Permutation Importance, TEST)")
    plt.xlabel("Importance (Δ PR-AUC when permuted)")
    plt.tight_layout()
    plt.show()

    print("\nMANAGEMENT INTERPRETATION:")
    print("- Threshold was chosen on validation data to hit a target recall for breaches,")
    print("  so the model acts like an early-warning tool (catch more breaches) rather than a strict classifier.")
    print("- PR-AUC is highlighted because breaches are rare; accuracy can be misleading.")
    print("=" * 90)

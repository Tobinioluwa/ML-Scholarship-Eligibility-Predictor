"""
Trains the scholarship eligibility classifier and saves the fitted
scikit-learn Pipeline (preprocessing + model) for use by the Flask app.
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/scholarship_data.csv"
MODEL_PATH = "model/scholarship_pipeline.joblib"
METRICS_PATH = "model/metrics.json"
FEATURE_IMPORTANCE_PATH = "static/images/feature_importance.png"

NUMERIC_FEATURES = [
    "age",
    "gpa",
    "family_income",
    "household_size",
    "extracurricular_score",
    "community_service_hours",
    "attendance_rate",
]
CATEGORICAL_FEATURES = [
    "gender",
    "has_disability",
    "is_first_generation",
    "previous_scholarship",
    "region",
]
TARGET = "eligible"


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def get_feature_names(pipeline: Pipeline) -> list:
    preprocessor = pipeline.named_steps["preprocess"]
    cat_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return NUMERIC_FEATURES + cat_names


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("Evaluation on held-out test set:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Not Eligible", "Eligible"]))

    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance chart for the "About the Model" page
    importances = pipeline.named_steps["model"].feature_importances_
    feature_names = get_feature_names(pipeline)
    order = np.argsort(importances)[::-1][:10]

    plt.figure(figsize=(8, 5))
    plt.barh(
        [feature_names[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#4f46e5",
    )
    plt.xlabel("Relative importance")
    plt.title("Top factors driving eligibility predictions")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=150)
    plt.close()

    print(f"\nSaved pipeline -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved feature importance chart -> {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()

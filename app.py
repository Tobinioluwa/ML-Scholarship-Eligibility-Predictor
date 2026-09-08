"""
Flask web application for the ML Scholarship Eligibility Predictor.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

MODEL_PATH = "model/scholarship_pipeline.joblib"
METRICS_PATH = "model/metrics.json"

app = Flask(__name__)

_pipeline = None
_metrics = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model file not found. Run `python src/data_generator.py` then "
                "`python src/train_model.py` to generate the dataset and train the model."
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def get_metrics():
    global _metrics
    if _metrics is None and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
    return _metrics or {}


FORM_FIELDS = [
    "age",
    "gender",
    "gpa",
    "family_income",
    "household_size",
    "extracurricular_score",
    "community_service_hours",
    "attendance_rate",
    "has_disability",
    "is_first_generation",
    "previous_scholarship",
    "region",
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    try:
        applicant = {
            "age": int(form["age"]),
            "gender": form["gender"],
            "gpa": float(form["gpa"]),
            "family_income": float(form["family_income"]),
            "household_size": int(form["household_size"]),
            "extracurricular_score": float(form["extracurricular_score"]),
            "community_service_hours": float(form["community_service_hours"]),
            "attendance_rate": float(form["attendance_rate"]),
            "has_disability": form["has_disability"],
            "is_first_generation": form["is_first_generation"],
            "previous_scholarship": form["previous_scholarship"],
            "region": form["region"],
        }
    except (KeyError, ValueError) as exc:
        return render_template("index.html", error=f"Invalid input: {exc}")

    pipeline = get_pipeline()
    X = pd.DataFrame([applicant])
    proba = float(pipeline.predict_proba(X)[0, 1])
    eligible = proba >= 0.5

    return render_template(
        "result.html",
        applicant=applicant,
        eligible=eligible,
        probability=round(proba * 100, 1),
    )


@app.route("/about")
def about():
    return render_template("about.html", metrics=get_metrics())


if __name__ == "__main__":
    app.run(debug=True)

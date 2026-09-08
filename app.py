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
                "Model file not found. Run `python src/prepare_dataset.py` then "
                "`python src/train_model.py` to build the dataset and train the model."
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
    "region",
    "household_size",
    "mother_education",
    "father_education",
    "weekly_study_time",
    "past_failures",
    "school_support",
    "family_support",
    "paid_tutoring",
    "internet_access",
    "extracurricular_activities",
    "absences",
    "prior_grade_1",
    "prior_grade_2",
    "current_grade",
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
            "region": form["region"],
            "household_size": form["household_size"],
            "mother_education": int(form["mother_education"]),
            "father_education": int(form["father_education"]),
            "weekly_study_time": int(form["weekly_study_time"]),
            "past_failures": int(form["past_failures"]),
            "school_support": form["school_support"],
            "family_support": form["family_support"],
            "paid_tutoring": form["paid_tutoring"],
            "internet_access": form["internet_access"],
            "extracurricular_activities": form["extracurricular_activities"],
            "absences": int(form["absences"]),
            "prior_grade_1": float(form["prior_grade_1"]),
            "prior_grade_2": float(form["prior_grade_2"]),
            "current_grade": float(form["current_grade"]),
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

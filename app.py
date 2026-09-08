"""
Flask web application for the ML Scholarship Eligibility Predictor.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd
from flask import Flask, render_template, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import scholarship_scraper  # noqa: E402

MODEL_PATH = "model/scholarship_pipeline.joblib"
METRICS_PATH = "model/metrics.json"
OPPORTUNITIES_CACHE_MAX_AGE = timedelta(hours=12)

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


def _cache_is_fresh(cache: dict) -> bool:
    try:
        generated_at = datetime.fromisoformat(cache["generated_at"])
    except (KeyError, ValueError, TypeError):
        return False
    return datetime.now(timezone.utc) - generated_at < OPPORTUNITIES_CACHE_MAX_AGE


def get_opportunities_data(force_refresh: bool = False):
    """Returns (data, error_message). data is always a dict with a
    'listings' key (possibly empty); error_message is set only when a
    refresh was attempted and failed with nothing usable to fall back on."""
    cache = scholarship_scraper.load_cache()

    if not force_refresh and cache and cache.get("listings") and _cache_is_fresh(cache):
        return cache, None

    try:
        fresh = scholarship_scraper.scrape_all()
        scholarship_scraper.save_cache(fresh)
        if not fresh["listings"]:
            return fresh, "No opportunities could be fetched right now — the sources may be temporarily unreachable. Please try again shortly."
        return fresh, None
    except Exception:
        if cache:
            return cache, "Couldn't refresh opportunities right now — showing the last saved results."
        return {"generated_at": None, "sources": [], "listings": []}, (
            "Couldn't load opportunities right now. Please try again in a few minutes."
        )


@app.route("/opportunities")
def opportunities():
    force_refresh = request.args.get("refresh") == "1"
    data, error = get_opportunities_data(force_refresh=force_refresh)
    listings = data.get("listings", [])

    all_levels = sorted({lvl for item in listings for lvl in item.get("level", [])})
    all_regions = sorted({r for item in listings for r in item.get("region", [])})

    q = request.args.get("q", "").strip().lower()
    level = request.args.get("level", "")
    region = request.args.get("region", "")
    funded_only = request.args.get("funded") == "1"

    filtered = listings
    if q:
        filtered = [
            it for it in filtered
            if q in it["title"].lower() or q in it["summary"].lower()
        ]
    if level:
        filtered = [it for it in filtered if level in it.get("level", [])]
    if region:
        filtered = [it for it in filtered if region in it.get("region", [])]
    if funded_only:
        filtered = [it for it in filtered if it.get("fully_funded")]

    return render_template(
        "opportunities.html",
        listings=filtered,
        total_count=len(listings),
        filtered_count=len(filtered),
        all_levels=all_levels,
        all_regions=all_regions,
        selected_q=request.args.get("q", ""),
        selected_level=level,
        selected_region=region,
        selected_funded=funded_only,
        generated_at=data.get("generated_at"),
        sources=data.get("sources", []),
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)

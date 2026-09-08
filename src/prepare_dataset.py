"""
Builds the training dataset from REAL, publicly available student records
(no synthetic/fabricated rows) — see data/raw/SOURCES.md for exact sources,
citations, and licenses.

Primary data: UCI "Student Performance" dataset (395 real students,
data/raw/student-mat.csv). We select a subset of its real columns that are
plausible for a scholarship applicant to self-report, and drop columns that
are irrelevant or overly invasive for an eligibility-screening context
(e.g. alcohol consumption, romantic relationship status, going-out
frequency, specific school name).

IMPORTANT — why the label is derived, not observed:
No public dataset records real students' actual "scholarship eligible"
outcome (that's sensitive financial/personal data tied to real people, so
it isn't released). So `eligible` is computed here with a fully documented,
deterministic rule applied to the real feature values below — it is NOT
random or fabricated, but it is also not a ground-truth label collected
from an actual scholarship committee. Treat this project's accuracy numbers
as a demonstration of the pipeline, not a validated real-world policy.

Rule (weights sum to 1.0, applied per-student, then top ~40% by score are
labeled eligible):

  merit_score  = weighted average of G1/G2/G3 (recent grades weighted more)
                 minus a penalty per past class failure
                 plus a small bonus for weekly study time
  need_score   = higher when parents' average education level is lower,
                 when there is no home internet access, when the student
                 does not receive paid tutoring, when family size is larger
                 (>3), and when there is no family educational support
  final_score  = 0.55 * merit_score + 0.35 * need_score
                 + 0.05 * (has extracurricular activities)
                 + 0.05 * (below-median absences)

This mirrors common real-world scholarship criteria (merit + financial/
social need) while being fully transparent about being a derived label.
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/student-mat.csv"
OUTPUT_PATH = "data/scholarship_data.csv"


def _minmax(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def build_dataset(raw_path: str = RAW_PATH) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)

    df = pd.DataFrame(
        {
            "age": raw["age"],
            "gender": raw["sex"].map({"F": "Female", "M": "Male"}),
            "region": raw["address"].map({"U": "Urban", "R": "Rural"}),
            "household_size": raw["famsize"].map(
                {"LE3": "3 or fewer", "GT3": "More than 3"}
            ),
            "mother_education": raw["Medu"],  # 0=none .. 4=higher education
            "father_education": raw["Fedu"],  # 0=none .. 4=higher education
            "weekly_study_time": raw["studytime"],  # 1=<2h .. 4=>10h
            "past_failures": raw["failures"],  # count of past class failures
            "school_support": raw["schoolsup"].map({"yes": "Yes", "no": "No"}),
            "family_support": raw["famsup"].map({"yes": "Yes", "no": "No"}),
            "paid_tutoring": raw["paid"].map({"yes": "Yes", "no": "No"}),
            "internet_access": raw["internet"].map({"yes": "Yes", "no": "No"}),
            "extracurricular_activities": raw["activities"].map(
                {"yes": "Yes", "no": "No"}
            ),
            "absences": raw["absences"],
            "prior_grade_1": raw["G1"],  # first period grade, 0-20
            "prior_grade_2": raw["G2"],  # second period grade, 0-20
            "current_grade": raw["G3"],  # most recent grade, 0-20
        }
    )

    # --- Merit score ---
    grade_avg = 0.2 * raw["G1"] + 0.3 * raw["G2"] + 0.5 * raw["G3"]
    merit_score = _minmax(grade_avg.to_numpy())
    merit_score = merit_score - 0.05 * raw["failures"].to_numpy()
    merit_score = merit_score + 0.05 * _minmax(raw["studytime"].to_numpy())
    merit_score = np.clip(merit_score, 0, None)

    # --- Need score ---
    parent_edu_avg = (raw["Medu"] + raw["Fedu"]) / 2
    parent_edu_need = 1 - _minmax(parent_edu_avg.to_numpy())
    no_internet = (raw["internet"] == "no").astype(float).to_numpy()
    no_paid_tutoring = (raw["paid"] == "no").astype(float).to_numpy()
    large_family = (raw["famsize"] == "GT3").astype(float).to_numpy()
    no_family_support = (raw["famsup"] == "no").astype(float).to_numpy()

    need_score = (
        0.55 * parent_edu_need
        + 0.15 * no_internet
        + 0.10 * no_paid_tutoring
        + 0.10 * large_family
        + 0.10 * no_family_support
    )

    # --- Small bonuses ---
    activity_bonus = (raw["activities"] == "yes").astype(float).to_numpy()
    low_absences_bonus = (raw["absences"] <= raw["absences"].median()).astype(
        float
    ).to_numpy()

    final_score = (
        0.55 * merit_score
        + 0.35 * need_score
        + 0.05 * activity_bonus
        + 0.05 * low_absences_bonus
    )

    threshold = np.quantile(final_score, 0.60)  # ~40% labeled eligible
    df["eligible"] = (final_score >= threshold).astype(int)

    return df


if __name__ == "__main__":
    dataset = build_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Built dataset from {RAW_PATH} ({len(dataset)} real student records) -> {OUTPUT_PATH}")
    print(dataset["eligible"].value_counts(normalize=True))

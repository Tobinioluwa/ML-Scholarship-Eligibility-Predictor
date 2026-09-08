"""
Synthetic dataset generator for the Scholarship Eligibility Predictor.

Why synthetic data?
--------------------
There is no widely available, download-without-authentication, applicant-level
dataset for "scholarship eligibility" (public datasets under this name are
either lists of scholarship *programs*, or require a Kaggle account/API key).
To keep this project 100% free, reproducible, and runnable offline, we
generate a realistic synthetic dataset from a documented rule + random noise.

The rule below encodes common real-world scholarship criteria:
academic merit, financial need, extracurricular engagement/community service,
attendance, and priority bonuses for first-generation students and students
with disabilities. Swap this script out for your own institution's historical
data (keeping the same column names) to train on real records instead.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 6000
OUTPUT_PATH = "data/scholarship_data.csv"


def _minmax(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + 1e-9)


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(16, 26, size=n_samples)
    gender = rng.choice(["Male", "Female", "Other"], size=n_samples, p=[0.48, 0.48, 0.04])

    gpa = np.clip(rng.normal(3.0, 0.55, size=n_samples), 0.0, 4.0).round(2)

    family_income = np.clip(rng.gamma(shape=2.2, scale=22000, size=n_samples), 0, 250000).round(0)

    household_size = np.clip(rng.poisson(4, size=n_samples), 1, 10)

    extracurricular_score = np.clip(rng.normal(5.0, 2.2, size=n_samples), 0, 10).round(1)

    community_service_hours = np.clip(rng.exponential(40, size=n_samples), 0, 300).round(0)

    attendance_rate = np.clip(rng.normal(88, 8, size=n_samples), 50, 100).round(1)

    has_disability = rng.choice(["Yes", "No"], size=n_samples, p=[0.08, 0.92])

    is_first_generation = rng.choice(["Yes", "No"], size=n_samples, p=[0.35, 0.65])

    previous_scholarship = rng.choice(["Yes", "No"], size=n_samples, p=[0.15, 0.85])

    region = rng.choice(["Urban", "Suburban", "Rural"], size=n_samples, p=[0.45, 0.35, 0.20])

    # --- Documented rule-based eligibility score ---
    academic_component = _minmax(gpa)
    need_component = _minmax(-family_income)  # lower income -> higher need
    engagement_component = 0.6 * _minmax(extracurricular_score) + 0.4 * _minmax(community_service_hours)
    attendance_component = _minmax(attendance_rate)

    bonus = (
        0.06 * (has_disability == "Yes")
        + 0.05 * (is_first_generation == "Yes")
        + 0.03 * (household_size >= 6)
    )

    score = (
        0.35 * academic_component
        + 0.30 * need_component
        + 0.15 * engagement_component
        + 0.10 * attendance_component
        + bonus
    )

    # Add noise so the boundary isn't perfectly sharp (mirrors real-world variability
    # in committee decisions, discretionary funding, etc.)
    noise = rng.normal(0, 0.06, size=n_samples)
    score_noisy = score + noise

    threshold = np.quantile(score_noisy, 0.60)  # ~40% of applicants are eligible
    eligible = (score_noisy >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "gpa": gpa,
            "family_income": family_income,
            "household_size": household_size,
            "extracurricular_score": extracurricular_score,
            "community_service_hours": community_service_hours,
            "attendance_rate": attendance_rate,
            "has_disability": has_disability,
            "is_first_generation": is_first_generation,
            "previous_scholarship": previous_scholarship,
            "region": region,
            "eligible": eligible,
        }
    )
    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(dataset)} rows -> {OUTPUT_PATH}")
    print(dataset["eligible"].value_counts(normalize=True))

# Raw Data Sources

Both files in this folder are **real, publicly available datasets** — not
generated or fabricated. They're committed here as-is (unmodified) so the
project builds reproducibly without a network dependency.

## `student-mat.csv` — UCI Student Performance Data Set (primary training data)

- 395 real students from two Portuguese secondary schools, math course.
- Original source: UCI Machine Learning Repository —
  https://archive.ics.uci.edu/dataset/320/student+performance
- Direct download (no login required): the UCI repository issues a zip via
  its dataset page; a plain-CSV mirror of the same, unmodified file used
  here is at
  https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv
- Citation: P. Cortez and A. Silva. *Using Data Mining to Predict Secondary
  School Student Performance.* In A. Brito and J. Teixeira (eds.),
  Proceedings of 5th FUBUTEC 2008, pp. 5-12, Porto, Portugal, 2008, EUROSIS.
- License: CC BY 4.0 (per the UCI repository listing).

## `admission_predict.csv` — Graduate Admission Prediction (supplementary reference)

- 400 real (anonymized) graduate-school applicant records: GRE/TOEFL
  scores, CGPA, research experience, and actual admission chance.
- Original source: Kaggle, "Graduate Admission 2" dataset by Mohan S
  Acharya — https://www.kaggle.com/datasets/mohansacharya/graduate-admissions
- Direct download (no login required, same unmodified file) used here:
  https://raw.githubusercontent.com/divyansha1115/Graduate-Admission-Prediction/master/Admission_Predict.csv
- Citation: Mohan S Acharya, Asfia Armaan, Aneeta S Antony, *A Comparison
  of Regression Models for Prediction of Graduate Admissions*, IEEE
  International Conference on Computational Intelligence in Data Science,
  2019.
- Used only as supplementary, real-world context for the merit-scoring
  methodology (see the About page) — its applicant population (international
  graduate-school candidates) differs from the primary dataset's, so it is
  **not merged** into the training data; it is not used to train the deployed
  model.

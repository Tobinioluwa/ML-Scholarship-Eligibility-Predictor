# 🎓 ML Scholarship Eligibility Predictor

A free, fully open-source machine learning system that predicts a student's
likelihood of being eligible for a scholarship, complete with a clean web
interface.

- **Model**: Random Forest classifier (scikit-learn)
- **Interface**: Flask web app with a Bootstrap-based UI
- **Cost**: $0 — every dependency is free and open-source, and it runs entirely
  on your own machine

## Why a synthetic dataset?

There isn't a widely available, download-without-authentication dataset of
*individual applicant* scholarship eligibility records — the public datasets
under that name are either lists of scholarship *programs* (not applicants)
or require a Kaggle account/API key. To keep this project free, reproducible,
and runnable offline, `src/data_generator.py` generates a realistic synthetic
dataset from a **documented rule** (academic merit + financial need +
engagement + attendance + priority bonuses, plus random noise). The exact
formula is in that file's comments.

To use real data instead: replace `data/scholarship_data.csv` with your own
records using the same column names, then re-run the training step below.

## Features used

| Feature | Description |
|---|---|
| `age` | Applicant age |
| `gender` | Male / Female / Other |
| `gpa` | GPA on a 0.0–4.0 scale |
| `family_income` | Annual family income (USD) |
| `household_size` | Number of people in the household |
| `extracurricular_score` | Extracurricular engagement, 0–10 |
| `community_service_hours` | Community service hours in the past year |
| `attendance_rate` | School attendance percentage |
| `has_disability` | Yes / No |
| `is_first_generation` | First-generation college student, Yes / No |
| `previous_scholarship` | Previously received a scholarship, Yes / No |
| `region` | Urban / Suburban / Rural |

## Project structure

```
.
├── app.py                     # Flask web app
├── requirements.txt
├── data/
│   └── scholarship_data.csv   # Generated synthetic dataset
├── src/
│   ├── data_generator.py      # Creates the synthetic dataset
│   └── train_model.py         # Trains + evaluates + saves the model
├── model/
│   ├── scholarship_pipeline.joblib  # Trained scikit-learn Pipeline
│   └── metrics.json                 # Held-out test metrics
├── templates/                 # HTML (Jinja2) templates
└── static/                    # CSS + generated charts
```

## Setup & Usage

1. **Install dependencies** (Python 3.9+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. **Generate the dataset**:

   ```bash
   python src/data_generator.py
   ```

3. **Train the model**:

   ```bash
   python src/train_model.py
   ```

   This prints accuracy/precision/recall/F1/ROC-AUC on a held-out test set and
   saves the trained pipeline to `model/scholarship_pipeline.joblib`.

4. **Run the web app**:

   ```bash
   python app.py
   ```

   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser, fill
   in the applicant form, and click **Predict Eligibility**. Visit `/about`
   to see model performance metrics and the top factors driving predictions.

## How predictions work

1. The Flask app loads the saved scikit-learn `Pipeline` (preprocessing +
   model) once at startup.
2. Form input is converted into a single-row `DataFrame` with the same
   column names used in training.
3. The pipeline standardizes numeric features, one-hot encodes categorical
   features, and the Random Forest outputs a probability of eligibility.
4. A probability ≥ 50% is shown as "Likely Eligible"; otherwise "Likely Not
   Eligible" — along with the exact probability.

## Deploying for free

This repo is ready to deploy on [Render](https://render.com)'s free tier
(no credit card required). It already includes:

- `Procfile` — tells the host to run `gunicorn app:app` (a production server)
- `render.yaml` — a Blueprint file Render can auto-detect
- `runtime.txt` — pins the Python version

**Steps:**

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to [render.com](https://render.com) and sign in with your GitHub account.
3. Click **New +** → **Web Service**, and select this repository.
4. Render should auto-detect the settings from `render.yaml`. If asked manually:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free
5. Click **Create Web Service**. After the build finishes (a couple of
   minutes), Render gives you a public URL like
   `https://ml-scholarship-eligibility-predictor.onrender.com` — open it to
   use the live app.

Note: on Render's free tier, the app "sleeps" after ~15 minutes of
inactivity and takes ~30–60 seconds to wake up on the next visit — this is
normal for free hosting and costs nothing.

Railway and PythonAnywhere work similarly and are also free-tier friendly.
No paid APIs or services are used anywhere in this project.

## Limitations & Disclaimer

This tool is trained on synthetic data for demonstration purposes and should
**not** be used as the sole basis for real financial-aid or scholarship
decisions. Always pair automated screening with human review.

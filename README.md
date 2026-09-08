# 🎓 ML Scholarship Eligibility Predictor

A free, fully open-source machine learning system that predicts a student's
likelihood of being eligible for a scholarship, complete with a clean web
interface.

- **Model**: Random Forest classifier (scikit-learn)
- **Interface**: Flask web app with a Bootstrap-based UI
- **Cost**: $0 — every dependency is free and open-source, and it runs entirely
  on your own machine

## The data — real, not synthetic

This project trains on **real student records**, not generated data:

- **Primary training data**: [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
  — 395 real students from two Portuguese secondary schools. Direct CSV
  (no login required): https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv
- **Supplementary reference only**: [Graduate Admission Prediction dataset](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)
  — 400 real applicant records, used only to produce a supporting real-world
  chart on the `/about` page (not merged into training data). Direct CSV:
  https://raw.githubusercontent.com/divyansha1115/Graduate-Admission-Prediction/master/Admission_Predict.csv

Both raw files are committed unmodified at `data/raw/`, with full citations
and licenses in `data/raw/SOURCES.md`.

**No public dataset records real students' actual scholarship decisions**
(that's sensitive personal/financial data, so it's never released). So while
every row is a real student, the `eligible` label is computed with a fully
documented, deterministic rule (merit from real grades + need from real
socioeconomic proxies — see `src/prepare_dataset.py` and the `/about` page
for the exact formula) — it is not an actual historical committee decision.
See **Limitations** below.

To retrain on your own institution's real historical data instead: replace
`data/raw/student-mat.csv` with your own records (matching columns), then
re-run the steps below.

## Features used

| Feature | Description |
|---|---|
| `age` | Student age |
| `gender` | Male / Female |
| `region` | Urban / Rural |
| `household_size` | "3 or fewer" / "More than 3" people |
| `mother_education`, `father_education` | 0 (none) – 4 (higher education) |
| `weekly_study_time` | 1 (<2h) – 4 (>10h) |
| `past_failures` | Count of past class failures |
| `school_support`, `family_support` | Yes / No |
| `paid_tutoring` | Receives paid extra tutoring, Yes / No |
| `internet_access` | Internet access at home, Yes / No |
| `extracurricular_activities` | Yes / No |
| `absences` | Number of school absences |
| `prior_grade_1`, `prior_grade_2`, `current_grade` | Grades on a 0–20 scale |

## Project structure

```
.
├── app.py                     # Flask web app
├── requirements.txt
├── data/
│   ├── raw/                   # Real, unmodified source datasets + SOURCES.md
│   ├── scholarship_data.csv   # Built dataset (real rows + derived label)
│   └── opportunities_cache.json  # Scraped listings cache (generated, gitignored)
├── src/
│   ├── prepare_dataset.py     # Builds the dataset from real data (documented label rule)
│   ├── train_model.py         # Trains + evaluates + saves the model
│   └── scholarship_scraper.py # Scrapes latest opportunities (see below)
├── model/
│   ├── scholarship_pipeline.joblib  # Trained scikit-learn Pipeline
│   └── metrics.json                 # Held-out test + cross-validation metrics
├── templates/                 # HTML (Jinja2) templates
└── static/                    # CSS + generated charts
```

## Live scholarship opportunities (`/opportunities`)

A second page shows currently open scholarships/fellowships pulled from
public RSS feeds, with filters and a direct link to apply on each listing.

**How it works:**
- `src/scholarship_scraper.py` fetches **RSS feeds only** (never raw HTML)
  from a small, editable list of global scholarship-news sites
  ([OpportunityDesk](https://opportunitydesk.org/), [Scholars4Dev](https://www.scholars4dev.com/)) —
  RSS is explicitly meant for syndication, so this is a much more stable and
  respectful way to pull "latest opportunities" than scraping arbitrary HTML.
- `robots.txt` is checked before every fetch; a source is skipped if it
  disallows the feed path.
- Each entry's study level (PhD, Master's, etc.), region, "fully funded"
  status, and any stated deadline are extracted with simple keyword/regex
  matching over the title and summary — **best-effort, not authoritative**.
  Every card links straight back to the original source so you can verify
  the real deadline and requirements before applying.
- Results are cached to `data/opportunities_cache.json` for 12 hours so the
  page stays fast and doesn't hammer the source sites on every visit; click
  **Refresh now** on the page (or visit `/opportunities?refresh=1`) to force
  an immediate re-fetch. If a fetch fails, the page falls back to the last
  successful results (or a friendly message if there are none yet) instead
  of crashing.
- Filters (search text, study level, region, fully-funded-only) are applied
  server-side via query parameters, so results are shareable/linkable.

To add more sources, add `{name, homepage, feed_url}` to the `SOURCES` list
in `src/scholarship_scraper.py` — no other code changes needed. To refresh
the cache manually from the command line: `python src/scholarship_scraper.py`.

## Setup & Usage

1. **Install dependencies** (Python 3.9+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. **Build the dataset** (from the real, committed source data):

   ```bash
   python src/prepare_dataset.py
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
- `.python-version` — pins the Python version (Render reads this, not the older Heroku-style `runtime.txt`)

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

- The `eligible` label is a documented derived rule, not a real committee's
  historical decision — no such public dataset exists for privacy reasons.
- Only 395 training rows from two Portuguese secondary schools — this won't
  generalize to other regions or scholarship programs without retraining on
  local data.
- Because the label is partly a function of grades, and grades are also
  model inputs, reported accuracy is optimistic relative to a truly
  independent, real-world labeled dataset.
- This tool should **not** be used as the sole basis for real financial-aid
  or scholarship decisions. Always pair automated screening with human
  review.
- The `/opportunities` page's deadline, level, and region tags are
  auto-extracted from third-party listing text and may be incomplete, stale,
  or wrong — always confirm details on the linked source page before
  applying to anything.

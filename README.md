# Credit default model — calibrated XGBoost

A portfolio project that began from a YouTube tutorial and was extended with a research interface, held-out re-evaluation of the shipped model, threshold and calibration analysis, a sensitivity tool and documented limitations. In short: an XGBoost pipeline, tuned with randomized search and wrapped in sigmoid
calibration, that estimates the probability that a loan in a public credit-risk dataset defaults. It is served by
FastAPI with an HTML/CSS/JavaScript applied-ML interface.

> Research demonstration only. Outputs are not credit scores, lending decisions or financial advice. The dataset's
> origin, period and currency are undocumented, and no fairness audit has been performed.

## Results on the held-out test split (shipped artifact)

`evaluation/evaluate_artifact.py` reproduces the notebook's cleaning and stratified 80/20 split
(`random_state=42`; 6,305 test rows, 1,362 defaults, identical to the notebook) and scores `credit_risk_model.pkl`
exactly as `main.py` does.

| Metric | Value |
|---|---|
| ROC-AUC | 0.953 |
| PR-AUC (average precision) | 0.910 |
| Brier score | 0.052 |
| At stored threshold 0.695: precision / recall / F1 | 0.967 / 0.742 / 0.840 |

The UI also shows the PR and ROC curves, a calibration curve, a threshold explorer with confusion matrices,
permutation importance, and the metrics recorded in the notebook.

## Interface

- **Assess:** a form with exactly the 11 `LoanApplication` fields. Categories and ranges come from the training data
  (`static/data/schema.json`); impossible values are blocked, and out-of-range values are allowed with an
  extrapolation warning. The loan-to-income ratio is computed as amount ÷ income (the dataset's definition) unless
  overridden.
- **Output:** probability, position relative to the stored threshold, the API's `Result` label, and the observed
  default rate among test applications that received similar scores.
- **Sensitivity:** varies one input across its training range and calls `POST /predict` at each value (an ICE curve).

## Architecture

```
static/
  index.html
  css/core.css, js/core.js   shared design system (synced from ../../_design_system)
  css/app.css, js/app.js     project UI
  data/schema.json           feature categories/ranges (generated)
  data/evaluation.json       held-out evaluation (generated)
main.py                      EXISTING FastAPI app (unchanged)
credit_risk_model.pkl        EXISTING calibrated pipeline (unchanged)
best_threshold.pkl           EXISTING decision threshold (unchanged)
evaluation/evaluate_artifact.py   NEW: regenerates static/data/*.json
Credit_Risk.ipynb            training notebook (unchanged)
```

No backend change was needed: the UI calls the existing `POST /predict`. Service status uses `GET /openapi.json`
because the API has no health route (the previous UI did the same). There was no Streamlit code in this project.

## Run locally

`requirements.txt` pins `xgboost==3.4.1`, which requires **Python ≥ 3.12**.

```bash
cd ML/CREDIT_RISK_ML
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Open <http://localhost:8000>. Run from the project directory (`main.py` loads the `.pkl` files by relative path).

**macOS:** XGBoost needs the OpenMP runtime. Install it with `brew install libomp`. Alternatively, reuse the copy
bundled with scikit-learn:

```bash
export DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs"
```

Regenerate the evaluation and schema files with `python evaluation/evaluate_artifact.py`.

## Changes to deployment configuration

- **Python 3.12 runtime (approved fix).** `runtime.txt` pinned `python-3.11.9`, but `xgboost==3.4.1` requires
  Python ≥ 3.12, so the Render build could not install the requirements. `runtime.txt` now pins `python-3.12.14`,
  and `render.yaml` sets `PYTHON_VERSION=3.12.14` (the setting Render reads). This is the version the shipped model
  was tested with locally: all pinned requirements install and `POST /predict` returns the same output as before.

## Known issues (model-related, not changed)

1. **Threshold selection:** the notebook computes `2·(P − R)/(P + R)` instead of F1, searches on the test split, and
   uses the uncalibrated model's probabilities. The saved threshold (0.695) does not match the printed value (1.00).
   It is nonetheless close to the F1-optimal threshold for the shipped model.
2. **Calibration:** mid-range probabilities (≈ 0.4–0.75) overstate the observed default rate on the test split.

## Limitations

- Loan grade and interest rate are lender-assigned and encode the lender's own risk view.
- Age is an input; no fairness or disparate-impact analysis was performed.
- Rows with missing employment length were dropped during cleaning; missing interest rates are median-imputed.
- Single public dataset; no temporal or external validation.

## Deployment notes

- No API keys or secrets are needed.
- **Dataset not included.** `credit_risk_dataset.csv` (32,581 loans) is not in this repository because its origin
  and licence are undocumented. The app does not need it: it uses `credit_risk_model.pkl` and the pre-computed
  `static/data/*.json`. To retrain or regenerate the evaluation, place your copy of `credit_risk_dataset.csv` in
  the project root (columns as listed in `Credit_Risk.ipynb`).
- Unpickling requires matching `scikit-learn==1.6.1` and `xgboost==3.4.1` on Python 3.12.

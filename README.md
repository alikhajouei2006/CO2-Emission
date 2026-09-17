# CO2 Emission Prediction (Pre-Production Scenario)

Predicting a vehicle's CO2 emissions (g/km) from its **declared specifications alone** — before it is physically built and fuel-consumption tested — using tree-based regression models trained on the [CO2 Emissions by Vehicles (Canada)](#dataset) dataset.

## Table of Contents

- [Problem Framing](#problem-framing)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology Summary](#methodology-summary)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Dataset](#dataset)
- [License](#license)

## Problem Framing

This project simulates a **pre-production** deployment scenario: predicting CO2 emissions using only the vehicle specifications that are known before physical prototyping and standardized fuel-consumption testing (`Make`, `Vehicle Class`, `Engine Size`, `Cylinders`, `Transmission`, `Fuel Type`).

Fuel-consumption features (City/Highway/Combined), while highly correlated with the target, are only measurable *after* a physical test and are therefore excluded from the model — using them would leak post-production information into a pre-production prediction task. An earlier iteration of this project that included these features reached a near-perfect R² (~0.99); inspecting feature importance revealed this was driven by data leakage rather than genuine predictive signal, which motivated the current framing. See `notebooks/CO2_Emission.ipynb`, Section 1.4, for the full discussion.

## Results

Final model: **XGBoost Regressor**, selected from four tuned candidates (Random Forest, XGBoost, Extra Trees, Decision Tree) based on cross-validated RMSE on the training set. The test set was touched exactly once, for this model only.

| Metric | Train | Test |
|---|---|---|
| R² | 0.97 | 0.95 |
| MAE (g/km) | 7.49 | 9.31 |
| RMSE (g/km) | 10.67 | 13.16 |

The most influential feature by permutation importance is `Engine Size(L)`, followed by `Vehicle Class` and `Cylinders` — consistent with the physical relationship between engine displacement and fuel burned.

## Repository Structure

```
├── data/                  # Raw dataset (see data/README.md for source/license)
├── notebooks/              # End-to-end exploratory notebook (EDA, modeling, diagnostics)
├── src/co2_predictor/       # Modular, OOP implementation used by scripts/
├── scripts/                # Training entrypoint (scripts/run_training.py)
├── models/                  # Persisted fitted pipeline (generated, not committed)
├── figures/                 # Selected plots referenced in this README
├── tests/                   # Unit tests for src/co2_predictor
└── reports/                 # Rendered HTML export of the notebook
```

## Installation

```bash
git clone https://github.com/<your-username>/co2-emission-prediction.git
cd co2-emission-prediction
pip install -e .
```

## Usage

**Reproduce the full analysis:** open `notebooks/CO2_Emission.ipynb`.

**Retrain the model from the command line:**
```bash
python scripts/run_training.py --data-path data/raw/CO2_Emissions_Canada.csv
```

**Predict on new vehicle specifications:**
```python
from co2_predictor.inference.predictor import CO2Predictor
import pandas as pd

predictor = CO2Predictor(model_path="models/xgb_pipeline.joblib")

new_vehicle = pd.DataFrame([{
    "Make": "TOYOTA",
    "Vehicle Class": "COMPACT",
    "Engine Size(L)": 2.0,
    "Cylinders": 4,
    "Transmission": "AS8",
    "Fuel Type": "Z",
}])

predictor.predict(new_vehicle)
```

## Methodology Summary

1. **Data cleaning** — duplicate removal (~15% of raw records were exact duplicates).
2. **EDA** — target distribution, VIF, ANOVA, Mutual Information; used to justify every feature-scope decision below, not just to describe the data.
3. **Feature scope** — dropped `Model` (high cardinality: 2053 unique values for 6282 records) and all fuel-consumption columns (pre-production framing, see above).
4. **Feature engineering** — a single derived feature (`Cylinders / Engine Size`) was tested against a rejected alternative (`Cylinders × Engine Size`) via cross-validation before being added to the final pipeline.
5. **Model comparison** — four tree-based models, each tuned with `RandomizedSearchCV` over a shared `RepeatedKFold` strategy (5 folds × 3 repeats); linear baselines were evaluated but underperformed substantially.
6. **Model selection & evaluation** — selection based entirely on train/CV metrics; the test set is used exactly once, at the end, for the selected model only.

Full reasoning for every decision above is documented inline in `notebooks/CO2_Emission.ipynb`.

## Limitations

- **Feature scope**: `Model` was excluded despite the highest Mutual Information among categorical features, due to its high cardinality relative to dataset size.
- **Underrepresented manufacturers**: some manufacturers (e.g., Bugatti, SRT) have as few as 2 records; performance for these cannot be reliably estimated from the current test set.
- **Domain coverage**: the dataset covers only internal-combustion vehicles sold in Canada — no electric or hybrid powertrains, and no other markets.
- **Single train/test split**: reported test metrics reflect one specific random split rather than a distribution across multiple splits.

## Future Work

- Explore target/frequency encoding for `Model` to recover predictive signal without the dimensionality cost of one-hot encoding.
- Extend to electric and hybrid vehicles to assess generalization beyond the current dataset's scope.
- Evaluate across multiple random splits/seeds to obtain a confidence interval on test performance.
- Expand `tests/` coverage and add CI (GitHub Actions) to run tests on every push.

## Dataset

[CO2 Emissions by Vehicles (Canada)](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64) — Government of Canada Open Data. See `data/README.md` for the exact source link, retrieval date, and license terms.

## License

See [LICENSE](LICENSE).

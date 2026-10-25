# CO2 Emission Prediction (Pre-Production Scenario)

This project predicts a vehicle's CO2 emissions (g/km) from its declared specifications alone, before a prototype exists and before any fuel-consumption test has been run. The models are tree-based regressors trained on the [CO2 Emission by Vehicles](#dataset) dataset.

The test set was used once, for the final model. Model comparison and selection used only the training data and cross-validation.

An earlier version of this project used the fuel-consumption columns and reached R² ≈ 0.99. Those columns are measured after a vehicle is built and tested, so they would not be available at prediction time, and the score said little about the model. The current version drops them (see [Problem framing](#problem-framing)).

## Contents

- [Problem framing](#problem-framing)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Method](#method)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Dataset](#dataset)
- [License](#license)
- [Author](#author)

## Problem framing

The task is to estimate CO2 emissions using only what is known before a vehicle is built: `Make`, `Vehicle Class`, `Engine Size(L)`, `Cylinders`, `Transmission` and `Fuel Type`.

The fuel-consumption columns (city, highway, combined) correlate with the target at about 0.9 or higher, but they can only be measured on a physical vehicle. In the first version of the project the model with these columns scored R² ≈ 0.99. Checking feature importance showed that nearly all of that came from the fuel columns, so they were removed. Section 1.4 of `notebook/CO2 Emission.ipynb` has the full discussion.

## Results

The final model is an XGBoost regressor. It was chosen from four tuned candidates (Random Forest, XGBoost, Extra Trees, Decision Tree) by cross-validated RMSE on the training set.

| Metric      | Train | Test  |
| ----------- | ----- | ----- |
| R²          | 0.97  | 0.95  |
| MAE (g/km)  | 7.49  | 9.31  |
| RMSE (g/km) | 10.67 | 13.16 |

Permutation importance ranks `Engine Size(L)` first, then `Vehicle Class` and `Cylinders`.

<p align="center">
  <img src="figures/final-parity-plot-xgb.png" width="32%" alt="Predicted vs. actual CO2 emissions on the test set">
  <img src="figures/final-residual-plot-xgb.png" width="32%" alt="Residual plot on the test set">
  <img src="figures/final-permutation-importance-xgb.png" width="32%" alt="Permutation importance of the final model on the test set">
</p>

## Repository structure

```
├── data/                     # Raw dataset (source and license in data/README.md)
├── notebook/                 # Notebook with the EDA, modeling and diagnostics
├── reports/                  # HTML export of the notebook
├── src/co2_predictor/        # Package used by the scripts
├── scripts/                  # Training entry point (scripts/run_training.py)
├── models/                   # Fitted pipeline, created by run_training.py (not committed)
├── figures/                  # Plots used in this README
├── tests/                    # Unit tests for src/co2_predictor
├── pyproject.toml
└── LICENSE
```

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/alikhajouei2006/CO2-Emission
cd CO2-Emission
pip install -e .
```

To also install pytest and Jupyter, use `pip install -e ".[dev]"`.

## Usage

To go through the full analysis, open `notebook/CO2 Emission.ipynb`.

To retrain from the command line:

```bash
python scripts/run_training.py
```

The script tunes all candidates, picks the final model, evaluates it once on the test set, and saves the fitted pipeline to `models/<model name>_pipeline.joblib`. With the current data that file is `models/XGB_pipeline.joblib`. Since `models/` is not committed, run training once before using the predictor.

To predict for a new vehicle:

```python
from co2_predictor.inference.predictor import CO2Predictor
import pandas as pd

predictor = CO2Predictor(model_path="models/XGB_pipeline.joblib")

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

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The tests check duplicate removal, that no fuel-consumption column reaches the model, the `Cylinders / Engine Size` feature, unseen categories at prediction time, and saving and loading the pipeline. They run on a small in-memory dataset with a simple model, so they finish in seconds and do not rerun the hyperparameter search.

## Method

1. Data cleaning. About 15% of the raw rows were exact duplicates and were removed.
2. EDA. Target distribution, VIF, ANOVA and mutual information were used to decide which columns to keep.
3. Feature scope. `Model` was dropped because it has 2,053 unique values for 6,282 records. All fuel-consumption columns were dropped for the reason given above.
4. Feature engineering. One derived feature, `Cylinders / Engine Size`, was added. A competing feature, `Cylinders × Engine Size`, was tested with cross-validation and rejected.
5. Model comparison. Four tree-based models were tuned with `RandomizedSearchCV` on a shared `RepeatedKFold` (5 folds, 3 repeats). Linear baselines scored much lower.
6. Selection and evaluation. The final model was chosen from training and cross-validation metrics. The test set was used once, at the end.

The notebook gives the reasoning behind each step, including the cases where a column with a high score was left out.

## Limitations

- `Model` was excluded even though it has the highest mutual information among the categorical columns, because its cardinality is high relative to the dataset size.
- Some manufacturers (Bugatti, SRT) have only 2 records, so performance on them cannot be estimated reliably.
- The data covers vehicles with combustion fuel types (gasoline, diesel, ethanol, natural gas), including conventional hybrids. It has no battery-electric or plug-in hybrid vehicles, and the model should not be used for them.
- The reported test metrics come from one random split, not from repeated splits.

## Future work

- Try target or frequency encoding for `Model`, to recover its signal without a large one-hot encoding.
- Repeat the evaluation over several seeds to get a confidence interval on test performance.
- Add tests for the tuning and selection code, and run tests on each push with GitHub Actions.
- Let `scripts/run_training.py` take the dataset path as an argument. It is currently set in `config.py`.
- Look at electric and plug-in hybrid vehicles, which need a different measurement framework.

## Dataset

[CO2 Emission by Vehicles](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles) (Kaggle), compiled from [Government of Canada Open Data](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64). `data/README.md` has the source link, retrieval date, license terms and a table of the column abbreviations.

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The dataset has its own license, described in [`data/README.md`](data/README.md).

## Author

Ali Khajouei  
[GitHub](https://github.com/alikhajouei2006/) · [LinkedIn](https://www.linkedin.com/in/ali-khajouei-81730135b/)

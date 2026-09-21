# Dataset

**Name:** CO2 Emission by Vehicles

**Downloaded from (Kaggle):**
https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles

**Original source:** Government of Canada Open Data (the Kaggle page is a compiled version covering 7 years of this data):
https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64

**Retrieved on:** 2026-2-2

**Rows / columns (raw):** 7,385 rows × 12 columns

**License:** (https://opendatacommons.org/licenses/dbcl/1-0/) Note this is the license of this specific compiled copy; the underlying Government of Canada data is separately governed by Canada's Open Government Licence.

**Note:** This is separate from, and not covered by, this repository's own
`LICENSE` file (MIT), which applies only to the code in this repository.

## Column Abbreviation Reference

These are documented by the dataset publisher; several were not obvious from the raw column values alone and are worth keeping here rather than re-deriving them from scratch later.

**`Model` suffixes**

| Abbreviation | Meaning               |
| ------------ | --------------------- |
| 4WD / 4X4    | Four-wheel drive      |
| AWD          | All-wheel drive       |
| FFV          | Flexible-fuel vehicle |
| SWB          | Short wheelbase       |
| LWB          | Long wheelbase        |
| EWB          | Extended wheelbase    |

**`Transmission`**

| Code          | Meaning                     |
| ------------- | --------------------------- |
| A             | Automatic                   |
| AM            | Automated manual            |
| AS            | Automatic with select shift |
| AV            | Continuously variable       |
| M             | Manual                      |
| 3–10 (suffix) | Number of gears             |

**`Fuel Type`**

| Code | Meaning          |
| ---- | ---------------- |
| X    | Regular gasoline |
| Z    | Premium gasoline |
| D    | Diesel           |
| E    | Ethanol (E85)    |
| N    | Natural gas      |

There is no separate code for hybrid or electric vehicles. A conventional
(non-plug-in) hybrid — e.g. a row with `Model` containing "HYBRID" — still
burns gasoline and is coded under its actual fuel type (usually `Z`); its
CO2 figure is measured the same way as any combustion vehicle. Pure
battery-electric (BEV) and plug-in hybrid (PHEV) vehicles are not
represented in this dataset at all — see `notebooks/CO2_Emission.ipynb`,
Section 3.6.2 / 1.4, and the "Domain coverage" item in `README.md`'s
Limitations section, for why that matters for this project's scope.

**`Fuel Consumption`** columns are in litres per 100 km (L/100 km); the
combined rating is 55% city / 45% highway, also given in miles per
gallon (mpg).

**`CO2 Emissions(g/km)`** is tailpipe CO2 for combined city/highway driving — this is the project's target variable.

## Files

- `raw/CO2 Emissions_Canada.csv` — original, unmodified file as downloaded.
  All cleaning (duplicate removal, feature selection) happens in code
  (`src/co2_predictor/data/`, `src/co2_predictor/features/`), never by
  hand-editing this file.

## Reproducing

If `raw/CO2 Emissions_Canada.csv` is not committed to this repository,
download it from the Kaggle link above and place it at this exact path
before running `scripts/run_training.py` or the notebook.

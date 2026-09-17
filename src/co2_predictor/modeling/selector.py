"""
Corresponds to notebook Section 9 (Optimized Model Evaluation) and
Section 11 (Final Model Selection).

Two responsibilities:

1. `build_comparison_table` -- assembles the train-set metrics +
   Mean/Std CV RMSE + RMSE Gap table for all tuned models (Section
   9 + 11.1). IMPORTANT: per the fix you already made, this must
   NEVER touch X_test/y_test. Only the final, single call in
   inference/... no -- in scripts/run_training.py, on the ONE
   selected model, touches test data. If you find yourself passing
   X_test into anything in this file, stop -- that reintroduces the
   exact issue you caught and fixed in Section 11.1.

2. `select_final_model` -- encodes the criteria from Section 11.2 as
   an actual (documented, testable) decision function instead of
   prose + manual eyeballing of a styled dataframe. Doesn't have to
   be fully automatic (you can still keep a human "why we agree with
   the automatic pick" note in the training script's output/log),
   but the ranking logic itself should be code, not just markdown.
"""

import pandas as pd

from co2_predictor.evaluation.metrics import evaluate_model


def build_comparison_table(
    *, fitted_searches: dict, X_train, y_train, metrics: dict
) -> pd.DataFrame:
    """Build the Section 11.1 comparison table (train + CV columns only).

    Moved from: Section 9.1-9.4 + Section 11.1's dict-assembly cells.
    """
    try:
        results = {}
        for name, search in fitted_searches.items():
            results[name] = evaluate_model(
                model=search,
                X=X_train, 
                y=y_train, 
                metrics=metrics
            )

            best_idx = search.best_index_

            mean_cv_score = round(search.best_score_, 3)
            std_cv = search.cv_results_['std_test_score'][best_idx]
            rmse_gap = mean_cv_score - results[name]['RMSE'] 

            results[name]['Std CV RMSE'] = std_cv
            results[name]['Mean CV RMSE'] = mean_cv_score
            results[name]['RMSE Gap'] = rmse_gap

        return pd.DataFrame(results)
    except Exception as e:
        return e



def select_final_model(comparison_table: pd.DataFrame) -> str:
    """Apply the Section 11.2 criteria and return the winning model name.

    Moved from: Section 11.2 (prose) + Section 11.3 (manual decision).
    Primary criterion: lowest Mean CV RMSE. Ties/close calls broken by
    the secondary criteria listed in 11.2 -- decide how "close" is
    defined (e.g. within 1 Std CV RMSE of the best) rather than
    leaving it implicit.
    """
    compare = {}
    weights = {
     
    }


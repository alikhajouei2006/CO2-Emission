import pandas as pd
from ..evaluation.metrics import evaluate_model

def build_comparison_table(
    *, fitted_searches: dict, X_train, y_train, metrics: dict
) -> pd.DataFrame:
    results = {}
    for name, search in fitted_searches.items():
        results[name] = evaluate_model(
            model=search,
            X=X_train, 
            y=y_train, 
            metrics=metrics
        )

        best_idx = search.best_index_

        mean_cv_score = round(-search.best_score_, 3)
        std_cv = search.cv_results_['std_test_score'][best_idx]
        rmse_gap = mean_cv_score - results[name]['RMSE'] 

        results[name]['Std CV RMSE'] = std_cv
        results[name]['Mean CV RMSE'] = mean_cv_score
        results[name]['RMSE Gap'] = rmse_gap

    return pd.DataFrame(results)

def select_final_model(comparison_table: pd.DataFrame) -> str:
    return comparison_table['Mean CV RMSE'].idxmin()
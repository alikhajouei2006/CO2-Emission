"""
End-to-end training entrypoint. Equivalent to running the entire
notebook top to bottom (Sections 2 through 12), minus the EDA
(Section 3) and diagnostics (Section 10) plotting, which are
analysis-only and belong in a notebook, not in this script.

Outline (fill in using the classes/functions built above):

    1.  Load config (co2_predictor.config)
    2.  data.loader.CO2DataLoader -> load_clean()                 [Section 2]
    3.  features.selection.FeatureScope.apply()                   [1.4, 3.5.2, 3.6.1, 3.6.2]
    4.  train_test_split(..., stratify=df['Make'])                [Section 5]
    5.  features.engineering.build_column_preprocessor()          [Section 5]
    6.  modeling.pipeline_factory.build_pipeline()                [Section 5]
    7.  modeling.trainer.HyperparameterTuner.search_all()         [Section 8]
    8.  modeling.selector.build_comparison_table()                [Sections 9, 11.1]
        -- train + CV only, X_test/y_test must not appear yet.
    9.  modeling.selector.select_final_model()                    [Section 11.2-11.3]
    10. evaluation.metrics.evaluate_model() on X_test/y_test,     [Section 12.1-12.2]
        called EXACTLY ONCE, only on the winning model.
    11. persistence.save_pipeline() the winning fitted pipeline.

Diagnostics (Section 10, 12.3-12.5) and EDA (Section 3) are
deliberately NOT called from this script. Run them separately, e.g.
from a `notebooks/diagnostics_report.ipynb` that imports
evaluation.diagnostics and loads the artifacts this script produced.

Consider argparse for: --data-path, --output-dir, --n-iter (search
budget), and a --skip-search flag that loads previously persisted
search results instead of re-running ~40 minutes of
RandomizedSearchCV (see the note in modeling/trainer.py).
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()

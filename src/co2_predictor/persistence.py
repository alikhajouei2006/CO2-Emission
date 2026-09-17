"""
Not present in the original notebook -- this is new, required for
the notebook -> package transition. The notebook lived entirely in
one Python process's memory; a real package needs to survive between
processes (training runs once, inference happens later/elsewhere/
repeatedly).

Saves/loads the single fitted sklearn Pipeline object (preprocessor +
feature engineering + scaler + model, all bundled together as one
object, since that's what Section 5's build_pipeline produces). This
is the ONE artifact that scripts/run_training.py produces and that
inference/predictor.py consumes -- keep the save format (joblib is
the standard choice for sklearn objects, more robust than pickle for
numpy-heavy objects) consistent between the two.
"""

from pathlib import Path

from sklearn.pipeline import Pipeline


def save_pipeline(pipeline: Pipeline, path: str | Path) -> None:
    """Persist a fitted Pipeline to disk (joblib)."""
    raise NotImplementedError


def load_pipeline(path: str | Path) -> Pipeline:
    """Load a previously persisted fitted Pipeline."""
    raise NotImplementedError

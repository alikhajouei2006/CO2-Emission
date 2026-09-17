import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline


def save_pipeline(pipeline: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: str | Path) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No fitted pipeline found at {path}")
    return joblib.load(path)
"""
Corresponds to notebook Section 10 (Model Diagnostics) and Section
12.3-12.5 (Final Residual/Parity/Permutation Importance).

*** ANALYSIS-ONLY MODULE ***
Nothing in this file is called by scripts/run_training.py's critical
path, and NOTHING here is ever imported by inference/predictor.py.
These functions exist so that a notebook (the original one, or any
future report) can call into tested, reusable plotting code instead
of redefining plot_residual/plot_parity/etc. inline. If you ever find
yourself importing this module from inference/ code, that's a sign
something has gone wrong architecturally -- diagnostics plots have no
business running when a live prediction is served.

Reminder from the fix you already made: every function here defaults
to being called with whatever split the CALLER passes in. Sections
10.x call these with (X_train, y_train) deliberately -- see the
Section 10 markdown note already in the notebook for why. Section
12.3-12.5 call the SAME functions with (X_test, y_test), exactly
once, for the final model only. Don't bake "train" or "test" into
the function names or defaults -- keep them split-agnostic, exactly
like evaluate_model in metrics.py.
"""
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance

def plot_learning_curve(model, X, y, model_name: str, *, cv: int) -> matplotlib.figure.Figure:
    try:
        sizes, train_scores, validation_scores = learning_curve(
            estimator=model,
            X=X,
            y=y,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )

        fig, ax = plt.subpltos(figsize=(8, 6))
        ax.plot(sizes, -train_scores.mean(axis=1), color='blue', label='Train')
        ax.plot(sizes, -validation_scores.mean(axis=1), color='green', label='Validation')
        ax.set_title(f'Learning Curve - {model_name}')
        ax.legend()
        plt.tight_layout()
        plt.close()
        return fig

    except Exception as e:
        return e


def plot_residual(*, model, X, y, model_name: str) -> matplotlib.figure.Figure:
    try:
        predicted_y = model.predict(X)
        error = y - predicted_y
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        residual_range = (np.min(error), np.max(error))
        mean_res = np.mean(error)
        text_str = (
            f"MAE: {mae:.2f}\n"
            f"RMSE: {rmse:.2f}\n"
            f"Mean Residual: {mean_res:.2f}\n"
            f"Range: [{residual_range[0]:.1f}, {residual_range[1]:.1f}]"
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(predicted_y, error, c='steelblue')
        ax.axhline(0, c='red', lw=2)
        ax.set_title(f'Residual Plot - {model_name}', font='serif', fontsize=20, pad=10)
        ax.set_xlabel('$y$ $pred$', font='serif', fontsize=14)
        ax.set_ylabel('Residual ($y$ - $\hat{y}$)', font='serif', fontsize=14)
        plt.text(
            0.02,
            0.98, 
            text_str, 
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top', 
            bbox=dict(
                boxstyle='round', 
                facecolor='white', 
                alpha=0.8
            )
        )
        plt.tight_layout()
        plt.close()
        return fig
    except Exception as e:
        return e

def plot_parity(*, model, X, y, model_name: str) -> matplotlib.figure.Figure:
    try:
        predicted_y = model.predict(X)
        min_val = min(y.min(), predicted_y.min())
        max_val = max(y.max(), predicted_y.max())

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y, predicted_y, c='steelblue', alpha=0.4)
        ax.plot([min_val, max_val], [min_val, max_val], c='red', lw=2)
        ax.set_xlabel('$y$', font='serif', fontsize=15)
        ax.set_ylabel('$\hat{y}$', font='serif', fontsize=15)
        ax.set_title(f'Parity Plot - {model_name}', font='serif', fontsize=20, pad=10)
        plt.tight_layout()
        plt.close()
        return fig
    except Exception as e:
        return e


def plot_feature_importance(
    *, model, feature_names, model_name: str
) -> matplotlib.figure.Figure:
    """
    Note: relies on model.named_steps['model'].feature_importances_,
    which only exists for tree-based estimators -- this function is
    not generic over arbitrary sklearn models, document that.
    """
    try:
        estimator = model.named_steps['model']
        importances = estimator.feature_importances_
        importance = pd.Series(importances, index=feature_names).sort_values()

        fig, ax = plt.subplots(figsize=(20,20))
        importance.plot.barh(ax=ax)
        ax.set_title(f'Feature Importance - {model_name}', font='serif', fontsize=20, pad=10)
        ax.set_xlabel("Importance", font='serif', fontsize=15)
        plt.tight_layout()
        plt.close(fig)
        return fig
    
    except Exception as e:
        return e


def compute_permutation_importance(
    *, model, X, y, feature_names, n_repeats: int = 10
) -> pd.DataFrame:
    """Moved from: Section 10.5 / 12.5, `compute_permutation_importance`."""
    try:
        perm_importance = permutation_importance(
            model,
            X,
            y,
            scoring="neg_root_mean_squared_error",
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        perm_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": perm_importance.importances_mean,
            "Std": perm_importance.importances_std
        })
        
        perm_importance_df = perm_importance_df.sort_values(
            "Importance",
            ascending=False
        )

        return perm_importance_df.set_index('Feature', drop=True)
    
    except Exception as e:
        return e
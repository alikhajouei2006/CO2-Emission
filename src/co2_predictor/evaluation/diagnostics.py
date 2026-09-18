import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance

def plot_learning_curve(model, X, y, model_name: str, *, cv: int) -> matplotlib.figure.Figure:
    sizes, train_scores, validation_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sizes, -train_scores.mean(axis=1), color='blue', label='Train')
    ax.plot(sizes, -validation_scores.mean(axis=1), color='green', label='Validation')
    ax.set_title(f'Learning Curve - {model_name.title()}')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_residual(*, model, X, y, model_name: str, set_name: str='Train Set') -> matplotlib.figure.Figure:
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
    ax.set_title(f'Residual Plot - {model_name.title()} ({set_name.title()})', font='serif', fontsize=20, pad=10)
    ax.set_xlabel(r'$y$ $pred$', font='serif', fontsize=14)
    ax.set_ylabel(r'Residual ($y$ - $\hat{y}$)', font='serif', fontsize=14)
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
    return fig   

def plot_parity(*, model, X, y, model_name: str, set_name: str='Train Set') -> matplotlib.figure.Figure:
    predicted_y = model.predict(X)
    min_val = min(y.min(), predicted_y.min())
    max_val = max(y.max(), predicted_y.max())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, predicted_y, c='steelblue', alpha=0.4)
    ax.plot([min_val, max_val], [min_val, max_val], c='red', lw=2)
    ax.set_xlabel(r'$y$', font='serif', fontsize=15)
    ax.set_ylabel(r'$\hat{y}$', font='serif', fontsize=15)
    ax.set_title(f'Parity Plot - {model_name.title()} ({set_name.title()})', font='serif', fontsize=20, pad=10)
    plt.tight_layout()
    return fig
    


def plot_feature_importance(
    *, model, feature_names, model_name: str, set_name: str='Train Set'
) -> matplotlib.figure.Figure:
    """
    Note: relies on model.named_steps['model'].feature_importances_,
    which only exists for tree-based estimators - this function is
    not generic over arbitrary sklearn models.
    """
    estimator = model.named_steps['model']
    importances = estimator.feature_importances_
    importance = pd.Series(importances, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(20,20))
    importance.plot.barh(ax=ax)
    ax.set_title(f'Feature Importance - {model_name.title()} ({set_name.title()})', font='serif', fontsize=20, pad=10)
    ax.set_xlabel("Importance", font='serif', fontsize=15)
    plt.tight_layout()
    return fig

def compute_permutation_importance(
    *, model, X, y, feature_names, 
    n_repeats: int = 10, random_state: int = 42
) -> pd.DataFrame:
    perm_importance = permutation_importance(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    perm_importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": perm_importance.importances_mean,
            "Std": perm_importance.importances_std
        }
    )
    
    perm_importance_df = perm_importance_df.sort_values(
        "Importance",
        ascending=False
    )

    return perm_importance_df.set_index('Feature', drop=True)
    
def plot_permutaion(
        *, permutaion_importance, model_name: str, set_name: str='Train Set'
    ) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    permutaion_importance.drop('Std', axis=1).plot(kind='barh', ax=ax[0])
    ax[0].set_title('Importance', font='serif', fontsize=16)
    permutaion_importance.drop('Importance', axis=1).plot(kind='barh', ax=ax[1])
    ax[1].set_title('Standard Deviation ', font='serif', fontsize=16)
    plt.suptitle(f'Permutaion Importance - {model_name.title()} ({set_name.title()})', font='serif', fontsize=20)
    plt.tight_layout()
    return fig
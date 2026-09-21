from co2_predictor.data.loader import CO2DataLoader
from co2_predictor.features.selection import FeatureScope
from co2_predictor.features.engineering import build_column_preprocessor
from co2_predictor.config import (RANDOM_STATE, TARGET, CV_N_SPLITS, \
                                      CV_N_REPEATS, N_ITER, initial_configurations, \
                                      DATA_PATH, TEST_SIZE)
from co2_predictor.modeling.registry import TUNED_MODEL_REGISTRY
from co2_predictor.modeling.trainer import HyperparameterTuner
from co2_predictor.modeling.selector import build_comparison_table, select_final_model
from co2_predictor.evaluation.metrics import METRICS, evaluate_model
from co2_predictor.persistence import save_pipeline
from sklearn.model_selection import train_test_split, RepeatedKFold

def load_and_prepare_data(*, path):
    data_loader = CO2DataLoader(path)
    df = data_loader.load_clean()
    feature_scope = FeatureScope(
        redundant_fuel_columns=('Fuel Consumption Comb (mpg)',),
        high_cardinality_columns=('Model',),
        inference_unavailable_columns=(
            'Fuel Consumption City (L/100 km)',
            'Fuel Consumption Hwy (L/100 km)',
            'Fuel Consumption Comb (L/100 km)'
        )
    )
    df = feature_scope.apply(df)
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train,X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size=TEST_SIZE,
        stratify=df['Make'],
        random_state=RANDOM_STATE
    )
    numerical_features = X.select_dtypes(include='number').columns.to_list()
    categorical_features = X.select_dtypes(include='object').columns.to_list()
    preprocessor = build_column_preprocessor(
        numerical=numerical_features,
        categorical=categorical_features
    )
    return X_train, X_test, y_train, y_test, preprocessor

def train_candidates(*, X_train, y_train, preprocessor):
    rkf = RepeatedKFold(
        n_repeats=CV_N_REPEATS,
        n_splits=CV_N_SPLITS, 
        random_state=RANDOM_STATE
    )
    hyperparameter_tuner = HyperparameterTuner(
        cv=rkf,
        n_iter=N_ITER,
        random_state=RANDOM_STATE
    )

    search_results = hyperparameter_tuner.search_all(
        registry=TUNED_MODEL_REGISTRY,
        preprocessor=preprocessor,
        X=X_train,
        y=y_train
    )

    return search_results

def select_and_report(*, search_results, X_train, y_train, metrics):
    comparison_table = build_comparison_table(
        fitted_searches=search_results,
        X_train=X_train,
        y_train=y_train,
        metrics=metrics
    )
    print(comparison_table)
    final_model_name = select_final_model(comparison_table=comparison_table)
    return final_model_name

def evaluate_final(*, final_model, X_test, y_test, metrics):
    eval_results = evaluate_model(
        model=final_model, 
        X=X_test, 
        y=y_test, 
        metrics=metrics
    )
    return eval_results

def main() -> None:
    print('CO2 emission prediction module - under pre-production scenario')
    print('Installing initial configurations...')
    initial_configurations()
    print('Initial cofigurations were installed successfully.')
    print('Loading and preparing data...')
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare_data(path=DATA_PATH)
    print('Data loading and prepration were done successfully.')
    print('Training candidates...')
    search_results = train_candidates(
        X_train=X_train, 
        y_train=y_train, 
        preprocessor=preprocessor
    )
    print('Candidates were trained successfully.')
    print('Selecting final model according to stated critera...')
    final_model_name = select_and_report(
        search_results=search_results,
        X_train=X_train,
        y_train=y_train,
        metrics=METRICS
    )
    print(f'Final model was selected successfully. \nFinal model name: {final_model_name}')
    final_model = search_results[final_model_name].best_estimator_
    print('Evaluating final model...')
    final_evaluation_results = evaluate_final(
        final_model=final_model,
        X_test=X_test, 
        y_test=y_test, 
        metrics=METRICS
    )
    print('Final model was evaluated successfully. \nResults: ')
    for metric, res in final_evaluation_results.items():
        print(f'{metric}: {res}')

    print('Saving final model...')
    save_path = r"models/xgb_pipeline.joblib"
    save_pipeline(final_model, save_path)
    print('Final model were successfully saved at: {save_path}')

if __name__ == "__main__":
    main()

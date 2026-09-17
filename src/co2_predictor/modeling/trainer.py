from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from .pipeline_factory import build_pipeline

class HyperparameterTuner:
    def __init__(self, *, cv: RepeatedKFold, n_iter: int, random_state: int):
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state

    def search(self, *, pipeline, param_distributions: dict, X, y) -> RandomizedSearchCV:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring='neg_root_mean_squared_error',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=2
        )

        return search.fit(X=X, y=y)
        

    def search_all(self, *, registry: dict, preprocessor, X, y) -> dict:
        search_results = {}

        for model_name, specs in registry.items():

            pipeline = build_pipeline(
                preprocessor=preprocessor,
                model=specs['estimator'],
            )
            search_results[model_name] = self.search(
                pipeline=pipeline,
                param_distributions=specs['param_distributions'], 
                X=X,
                y=y
            )

        return search_results
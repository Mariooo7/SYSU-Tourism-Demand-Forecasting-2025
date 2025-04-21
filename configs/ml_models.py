from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

MODELS = {
    'lgbm': {
        'model': LGBMRegressor(verbose=-1),
        'params': {
            'num_leaves': [63, 127, 255],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [200, 500],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
    },
    'catboost': {
        'model': CatBoostRegressor(silent=True),
        'params': {
            'iterations': [200, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [5, 7],
            'l2_leaf_reg': [3, 5, 10]
        }
    },
    'random_forest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [300, 500],
            'max_depth': [5, 7, 10],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 5]
        }
    }
}

ENSEMBLE_MODEL = {
    'model': RandomForestRegressor(),
    'params': {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 5]
    }
}

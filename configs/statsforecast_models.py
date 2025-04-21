from statsforecast.models import AutoETS, AutoARIMA, AutoTBATS, DynamicOptimizedTheta, MSTL
from sklearn.ensemble import RandomForestRegressor

# 设置模型参数
# 设置模型参数
model_configs = [
    {
        'name': 'ETS',
        'class': AutoETS,
        'params': {
            'season_length': 7
        }
    },
    {
        'name': 'SARIMA',
        'class': AutoARIMA,
        'params': {
            'season_length':7
        }
    },
    {
        'name': 'AutoTBATS',
        'class': AutoTBATS,
        'params': {
           'season_length': 7
        }
    },
    {
        'name': 'DynamicOptimizedTheta',
        'class': DynamicOptimizedTheta,
        'params': {
            'season_length': 7
        }
    },
    {
        'name': 'MSTL',
        'class': MSTL,
        'params': {
            'season_length': 7
        }
    }
]

ensemble_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=5,
                random_state=42
                )
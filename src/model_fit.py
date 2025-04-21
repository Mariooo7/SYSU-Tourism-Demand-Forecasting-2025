from configs.ml_models import MODELS, ENSEMBLE_MODEL
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import numpy as np
from src.preprocessing import split_processed_data

def train_models(X_train, y_train, X_val, y_val):
    """
    训练多个模型并保存最优版本
    返回最优模型字典和集成模型
    """
    best_models = {}
    model_mape = {}

    # 训练单个模型
    for name in MODELS:
        model_configs = MODELS[name]
        model = model_configs['model']
        params = model_configs['params']
        # 随机搜索参数优化
        search = RandomizedSearchCV(model,
                                    params,
                                    cv=5,
                                    n_iter=100,
                                    scoring='neg_mean_absolute_percentage_error')
        search.fit(X_train, y_train)

        # 验证集评估
        best_models[name] = search.best_estimator_
        val_pred = best_models[name].predict(X_val)
        mape = mean_absolute_percentage_error(y_val, val_pred)
        model_mape[name] = mape  # 记录单个模型MAPE

    # 创建集成模型
    estimators = [(name, model) for name, model in best_models.items()]


    # 优化集成模型
    final_search = RandomizedSearchCV(
        ENSEMBLE_MODEL['model'],
        ENSEMBLE_MODEL['params'],
        cv=5,
        n_iter=100,
        scoring='neg_mean_absolute_percentage_error'
    )
    final_search.fit(X_train, y_train)

    # 使用最优参数构建最终估计器
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=final_search.best_estimator_
    )
    # 训练并保存集成模型
    stacking.fit(X_train, y_train)
    # 计算集成模型MAPE
    stacking_pred = stacking.predict(X_val)
    stacking_mape = mean_absolute_percentage_error(y_val, stacking_pred)
    model_mape['stacking'] = stacking_mape  # 记录集成模型MAPE

    return best_models, stacking, model_mape


def train_and_save_models(processed_data_dict, model_dir=None,
                          cumulative_thresholds=None, min_features=10):
    """
    训练模型并保存，支持自定义特征选择阈值

    参数:
        processed_data_dict: 处理后的数据字典 {名称: 处理后的数据}
        model_dir: 模型保存目录
        cumulative_thresholds: 各口岸累积重要性阈值字典
        min_features: 最小保留特征数

    返回:
        模型文件路径列表
    """
    models = []

    for name, processed_data in processed_data_dict.items():
        X_train, X_test, y_train, y_test = split_processed_data(processed_data)

        # 第一轮训练获取特征重要性
        best_models, stacking, mape = train_models(X_train, y_train, X_test, y_test)

        # 获取最优模型
        best_model_name = min(mape, key=mape.get)
        best_model = stacking if best_model_name == 'stacking' else best_models[best_model_name]

        if cumulative_thresholds is not None:
            # 特征重要性筛选
            if best_model_name == 'stacking':
                base_model = best_model.estimators_[0]
                importance = base_model.feature_importances_
                features = base_model.feature_names_in_
            elif hasattr(best_model, 'feature_importances_'):
                importance = best_model.feature_importances_
                if hasattr(best_model, 'feature_names_in_'):  # sklearn
                    features = best_model.feature_names_in_
                elif hasattr(best_model, 'feature_name_'):  # LightGBM
                    features = best_model.feature_name_
                elif hasattr(best_model, 'feature_names_'):  # CatBoost
                    features = best_model.feature_names_
                else:
                    features = X_train.columns.tolist()
                    print('未筛选特征，使用所有特征')
            elif hasattr(best_model, 'get_feature_importance'):
                importance = best_model.get_feature_importance()
                features = best_model.feature_names_
            else:
                features = X_train.columns.tolist()
                importance = np.ones(len(features))

            # 设置动态阈值
            sorted_idx = np.argsort(importance)[::-1]
            cumulative = np.cumsum(importance[sorted_idx])
            threshold = cumulative_thresholds.get(name, 0.9)  # 默认0.9
            threshold_idx = np.where(cumulative >= threshold * cumulative[-1])[0][0]
            selected_features = [features[i] for i in sorted_idx[:max(threshold_idx + 1, min_features)]]

            # 筛选特征后重新训练
            if len(selected_features) < len(features):
                X_train_selected = X_train[selected_features]
                X_test_selected = X_test[selected_features]
                best_models, stacking, mape = train_models(X_train_selected, y_train, X_test_selected, y_test)
                best_model_name = min(mape, key=mape.get)
                best_model = stacking if best_model_name == 'stacking' else best_models[best_model_name]
        else:
            # 不进行特征筛选
            selected_features = X_train.columns.tolist()

        # 保存模型
        filename = f"{model_dir}/{name}_best_{best_model_name}.pkl"
        joblib.dump({
            'model': best_model,
            'features': selected_features if 'selected_features' in locals() else X_train.columns.tolist(),
            'name': name,
            'mape': mape[best_model_name]
        }, filename)
        models.append(filename)

        print(f"{name} 最佳模型已保存：{filename} (MAPE={mape[best_model_name]:.4f})")
        print(f"特征选择：{selected_features}")
    return models

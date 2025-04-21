from statsforecast import StatsForecast
from configs.statsforecast_models import model_configs
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error as mape
import numpy as np
from configs.statsforecast_models import ensemble_model
from src.preprocessing import split_data

# 定义滚动窗口预测器
class RollingForecaster:
    def __init__(self, train_df, test_df,steps=None, model_configs=model_configs):
        self.train = train_df # 训练集
        self.test = test_df # 测试集
        self.steps = steps.copy() if steps else [1,2,3] # 预测步长
        self.model_configs = model_configs # 模型配置
        self.best_model = None # 最优模型
        self.best_model_name = None  # 新增最优模型名称属性

        # 动态创建模型实例
        self.models = [
            cfg['class'](**cfg['params'])  # 解包参数实例化
            for cfg in model_configs # 遍历模型配置
        ]
        self.model_names = [cfg['name'] for cfg in model_configs] # 模型名称

    def rolling_forecast(self, model, horizon):
        current_train = self.train # 定义当前训练集
        current_test = self.test
        predictions= [] # 定义预测结果
        # 循环预测
        for i in range(0, len(self.test), horizon):
            fcst = StatsForecast(
                models=[model],
                freq='D', # 时间频率(日)
            )
            forecast = fcst.forecast(
                df = current_train,
                h=horizon,
                time_col='Date',
                target_col='Total'
            )
            # 记录结果
            used_test = current_test[:horizon]
            pred = forecast[model.__class__.__name__].values[:len(used_test)]
            predictions.extend(pred)
            # 更新训练集和测试集
            current_train = pd.concat([current_train[horizon:], used_test])
            current_test = current_test[len(used_test):]
        return predictions

    def run(self):
        global ensemble_model
        metrics = pd.DataFrame(index=self.steps, columns=self.model_names.append('Ensemble'))
        df_predictions = self.test.copy()

        for h in self.steps:
            # 收集各模型预测结果（保持不变）
            model_preds = []
            for model, name in zip(self.models, self.model_names):
                results = self.rolling_forecast(model, h)
                df_predictions[f'{name}_{h}'] = results
                metrics.loc[h, name] = mape(self.test['Total'], results)
                model_preds.append(results)

            # ==== 新增集成模型训练逻辑 ====
            # 生成时间序列交叉验证数据
            X_train_stack = []
            y_train_stack = []
            window_size = h * 20  # 动态调整窗口大小
            val_size = h * 5 # 验证窗口移动步长

            for i in range(0, len(self.train) - window_size - val_size, val_size):
                # 使用前window_size数据训练基础模型
                train_window = self.train[i:i + window_size]
                val_window = self.train[i + window_size:i + window_size + val_size]

                # 生成基础模型预测
                base_preds = []
                for model in self.models:
                    fcst = StatsForecast(models=[model], freq='D')
                    pred = fcst.forecast(df=train_window, h=h*5, time_col='Date', target_col='Total')
                    base_preds.append(pred[model.__class__.__name__].values)

                # 堆叠预测结果作为特征
                X_window = np.column_stack(base_preds)
                X_train_stack.append(X_window)
                y_train_stack.extend(val_window['Total'].values)

            # 训练集成模型
            X_train = np.vstack(X_train_stack)
            y_train = np.array(y_train_stack)
            ensemble_model.fit(X_train, y_train)

            # 生成测试集预测
            X_test = np.column_stack(model_preds)
            ensemble_pred = ensemble_model.predict(X_test)
            # ==============================

            # 记录集成模型结果
            df_predictions[f'Ensemble_{h}'] = ensemble_pred
            metrics.loc[h, 'Ensemble'] = mape(self.test['Total'], ensemble_pred)

            # 更新最优模型选择逻辑
            current_mape = metrics.loc[h].dropna()
            self.best_model_name = current_mape.idxmin()

            if self.best_model_name == 'Ensemble':
                self.best_model = ensemble_model
            else:
                self.best_model = self.models[self.model_names.index(self.best_model_name)]

        return metrics, df_predictions, ensemble_model

    def rolling_predict(self, data, model, days, horizon):
        predictions= [] # 定义预测结果
        # 循环预测
        for i in range(0, days, horizon):
            fcst = StatsForecast(
                models=[model],
                freq='D', # 时间频率(日)
            )
            forecast = fcst.forecast(
                df = data,
                h=horizon,
                time_col='Date',
                target_col='Total'
            )
            # 记录结果
            pred = forecast[model.__class__.__name__].values[:min(days-i, horizon)]
            predictions.extend(pred)
            pred_df = pd.DataFrame({
                'Date': pd.date_range(start=data['Date'].iloc[-1], periods=len(pred) + 1, freq='D')[1:],
                'Total': pred
            })
            data = pd.concat([data, pred_df])
        return predictions

def process_portfolio(name, data, step):
    train, test = split_data(data)
    forecaster = RollingForecaster(train, test, steps=[step])
    metrics, predictions, ensemble_model = forecaster.run()

    # 选择最优模型
    avg_metrics = metrics.mean()
    forecaster.best_model_name = avg_metrics.idxmin()

    # 返回必要对象用于后续预测
    return {
        'name': name,
        'forecaster': forecaster,
        'ensemble_model': ensemble_model,
        'metrics': metrics,
        'data': data,
        'step': step
    }

def rolling_predict(data, model, days, horizon):
    predictions= [] # 定义预测结果
    # 循环预测
    current_data = data.copy()
    for i in range(0, days, horizon):
        fcst = StatsForecast(
            models=[model],
            freq='D', # 时间频率(日)
        )
        forecast = fcst.forecast(
            df = current_data,
            h=horizon,
            time_col='Date',
            target_col='Total'
        )
        # 记录结果
        pred = forecast[model.__class__.__name__].values[:min(days-i, horizon)]
        predictions.extend(pred)
        pred_df = pd.DataFrame({
            'unique_id': 1,
            'Date': pd.date_range(start=current_data['Date'].iloc[-1]+pd.Timedelta(days=1), periods=len(pred)),
            'Total': pred
        })
        current_data = pd.concat([current_data, pred_df])
    return predictions


def predict_future_results(result_dict, forecast_days=5):
    """
    使用训练好的模型进行未来预测

    参数:
        result_dict: 包含训练结果的字典 {名称: 训练结果}
        forecast_days: 预测天数，默认为5

    返回:
        包含预测结果的字典 {名称: (预测值数组, MAPE, 模型名称)}
    """
    predictions = {}

    for name, res in result_dict.items():
        # 解包存储的结果
        forecaster = res['forecaster']
        data = res['data']
        step = res['step']
        current_step = step
        metrics_min = res['metrics'].loc[current_step].min()

        # 准备数据
        data_with_id = data.copy()
        data_with_id.insert(0, 'unique_id', 1)

        # 进行预测
        if forecaster.best_model_name == 'Ensemble':
            ensemble_model = res['ensemble_model']
            models_preds = []
            for model in forecaster.models:
                model_pred = rolling_predict(data_with_id, model, days=forecast_days, horizon=current_step)
                models_preds.append(model_pred)
            X_stack = np.column_stack(models_preds)
            final_pred = ensemble_model.predict(X_stack)[:forecast_days]
        else:
            best_model = forecaster.models[forecaster.model_names.index(forecaster.best_model_name)]
            final_pred = rolling_predict(data_with_id, best_model, days=forecast_days, horizon=current_step)

        # 保存结果
        predictions[name] = (final_pred, metrics_min, forecaster.best_model_name)

        # 打印结果
        print(f'{name}: 未来{forecast_days}天预测结果 {final_pred}')
        print(f'MAPE: {metrics_min} MODEL: {forecaster.best_model_name}')

    return predictions


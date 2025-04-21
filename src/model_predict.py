import pandas as pd
import numpy as np
import joblib
from workalendar.asia import HongKong
import chinese_calendar as calendar

rf_data = pd.read_csv('data/weather/rf.csv', parse_dates=['Date'], date_format='%Y-%m-%d')

def forecast_future(data, model, steps=5, required_features=None):
    """
    生成未来特征并进行预测
    参数:
        data: 包含历史特征的DataFrame
        model: 训练好的预测模型
        steps: 预测步长
        required_features: 模型所需的特征列表
    返回:
        包含预测值的DataFrame
    """
    # 确保数据按时间顺序排序
    """
    递归生成未来特征并进行预测
    参数:
        data: 包含历史特征的DataFrame
        model: 训练好的预测模型
        steps: 预测步长
    返回:
        包含预测值的DataFrame
    """
    future_df = data.copy()

    for _ in range(steps):
        # 生成新时间步
        new_date = future_df['Date'].iloc[-1] + pd.Timedelta(days=1)
        new_row = {}
        new_row['Date'] = new_date

        # 基础时间特征
        new_row['year'] = new_date.year
        new_row['month'] = new_date.month
        new_row['day'] = new_date.day
        new_row['weekday'] = new_date.weekday()
        new_row['quarter'] =  (new_date.month - 1) // 3 + 1 # 季度计算
        new_row['is_weekend'] = new_row['weekday'] in (5, 6)

        # 香港节假日
        cal = HongKong()
        new_row['is_hk_holiday'] = not cal.is_working_day(new_date)
        new_row['is_cn_holiday'] = not calendar.is_workday(new_date)

        # 节假日前后特征
        new_row['pre_hk_holiday'] = future_df['is_hk_holiday'].iloc[-1]
        new_row['pre_cn_holiday'] = future_df['is_cn_holiday'].iloc[-1]
        new_row['post_hk_holiday'] = future_df['is_hk_holiday'].iloc[-1]
        new_row['post_cn_holiday'] = future_df['is_cn_holiday'].iloc[-1]
        new_row['hk_weekend_holiday'] = new_row['is_weekend'] * new_row['is_hk_holiday']
        new_row['cn_weekend_holiday'] = new_row['is_weekend'] * new_row['is_cn_holiday']

        new_baidu_columns = ['hk_sum','hk_pc','hk_mobile',
                     'hk_food_sum','hk_food_pc','hk_food_mobile',
                     'hk_hotel_sum','hk_hotel_pc','hk_hotel_mobile',
                     'hk_map_sum','hk_map_pc','hk_map_mobile',
                     'hk_metro_sum','hk_metro_pc','hk_metro_mobile',
                     'hk_shopping_sum','hk_shopping_pc','hk_shopping_mobile',
                     'hk_show_sum','hk_show_pc','hk_show_mobile',
                     'hk_weather_sum','hk_weather_pc','hk_weather_mobile',
                     'baidu_sum', 'baidu_pc', 'baidu_mobile' ,'pc_mobile']
        # 滞后特征（使用最后可用的历史/预测值）
        for lag in  [1, 2, 3, 4, 5, 7, 14, 21, 30]:
            new_row[f'lag_{lag}'] = future_df['Total'].iloc[-lag] if len(future_df) >= lag else np.nan
        for lag in  [5, 7, 14]:
            new_row[f'ex_rate_lag_{lag}'] = future_df['ex'].iloc[-lag] if len(future_df) >= lag else np.nan
            for col in new_baidu_columns:
                new_row[f'{col}_lag_{lag}'] = future_df[col].iloc[-lag] if len(future_df) >= lag else np.nan
        # 汇率变化特征
        new_row['ex_rate_change_14d'] = new_row['ex_rate_lag_7'] / new_row['ex_rate_lag_14'] - 1
        new_row['ex_rate_volatility_7d'] = future_df['ex_rate_lag_5'].iloc[-7:].std()
        # 降雨预报特征
        new_row['rf'] = rf_data[rf_data['Date'] == new_date]['rain'].values[0] if new_date in rf_data['Date'].values else np.nan
        # 滚动统计（基于历史窗口）
        new_row['rolling_7_mean'] = future_df['Total'].iloc[-7:].mean()
        new_row['rolling_7_std'] = future_df['Total'].iloc[-7:].std()
        new_row['rolling_30_mean'] = future_df['Total'].iloc[-30:].mean()
        new_row['ma_ratio_7_30'] = new_row['rolling_7_mean'] / new_row['rolling_30_mean']
        new_row['rolling_7_ex_rate_mean'] = future_df['ex_rate_lag_7'].iloc[-7:].mean()

        # 傅里叶项（月度周期）
        new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)

        # 趋势特征
        new_row['days_since_start'] = future_df['days_since_start'].iloc[-1] + 1
        new_row['days_squared'] = new_row['days_since_start'] ** 2

        # 同比/环比
        new_row['weekly_growth'] = new_row['lag_7'] / new_row['lag_14']

        # # 合并新数据
        new_row = pd.DataFrame([new_row], columns=future_df.columns)
        future_df = pd.concat([future_df, new_row])

        # 更新索引为日期
        future_df.index = future_df['Date']
        # 预测当前步
        current_features = future_df.iloc[[-1]]
        pred = model.predict(current_features[required_features])
        future_df.loc[new_date, 'Total'] = pred[0]


    return future_df.iloc[-steps:]


def predict_with_models(df_dict, models_list, forecast_days=5):
    """
    根据模型列表匹配数据字典中的数据进行预测

    参数:
        df_dict: 包含数据的字典 {名称: DataFrame}
        models_list: 模型路径列表
        forecast_days: 预测天数，默认为5

    返回:
        包含预测结果的字典 {名称: (预测值, MAPE)}
    """
    results = {}
    for (name, df), model_path in zip(df_dict.items(), models_list):
        saved_data = joblib.load(model_path)
        model = saved_data['model']
        required_features = saved_data['features']
        mape = saved_data['mape']

        prediction = forecast_future(df, model, forecast_days, required_features)['Total']
        results[name] = (prediction, mape)
        print(f"{name} 预测结果: {prediction} (MAPE={mape:.4f})")

    return results
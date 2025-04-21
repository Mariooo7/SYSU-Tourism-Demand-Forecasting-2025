import pandas as pd
import numpy as np
from workalendar.asia import HongKong
import chinese_calendar as calendar

ex_data = pd.read_excel('data/ex_rate.xlsx', parse_dates=['Date'], date_format='%d-%m-%Y')
baidu_data = pd.read_csv('data/baidu/baidu_df.csv', parse_dates=['Date'], date_format='%Y-%m-%d')
rf_data = pd.read_csv('data/weather/rf.csv', parse_dates=['Date'], date_format='%Y-%m-%d')

# 特征工程函数
def create_features(raw_df=pd.DataFrame()):
    df = raw_df.copy()
    # 合并汇率数据
    df = df.merge(ex_data, on='Date', how='left')
    df.columns = ['Date', 'Total', 'ex']
    # 合并百度指数数据
    df = df.merge(baidu_data, on='Date', how='left')
    # 百度指数特征处理
    baidu_columns = ['hk_sum','hk_pc','hk_mobile',
                     'hk_food_sum','hk_food_pc','hk_food_mobile',
                     'hk_hotel_sum','hk_hotel_pc','hk_hotel_mobile',
                     'hk_map_sum','hk_map_pc','hk_map_mobile',
                     'hk_metro_sum','hk_metro_pc','hk_metro_mobile',
                     'hk_shopping_sum','hk_shopping_pc','hk_shopping_mobile',
                     'hk_show_sum','hk_show_pc','hk_show_mobile',
                     'hk_weather_sum','hk_weather_pc','hk_weather_mobile']
    baidu_sum_columns = [col for col in baidu_columns if 'sum' in col]
    baidu_pc_columns = [col for col in baidu_columns if 'pc' in col]
    baidu_mobile_columns = [col for col in baidu_columns if 'mobile' in col]
    df['baidu_sum'] = df[baidu_sum_columns].sum(axis=1)
    df['baidu_pc'] = df[baidu_pc_columns].sum(axis=1)
    df['baidu_mobile'] = df[baidu_mobile_columns].sum(axis=1)
    df['pc_mobile'] = df['baidu_pc'] / (df['baidu_mobile'] + 1e-6)
    new_baidu_columns = baidu_columns + ['baidu_sum', 'baidu_pc', 'baidu_mobile' ,'pc_mobile']
    # 降雨特征
    df = df.merge(rf_data, on='Date', how='left')
    # 时间基础特征
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['weekday'] = df['Date'].dt.weekday
    df['quarter'] = df['Date'].dt.quarter
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    # 处理汇率周末缺失值
    df['ex'] = df['ex'].fillna(method='ffill')
    # 香港节假日标记
    cal = HongKong()
    df['is_hk_holiday'] = df['Date'].apply(lambda x: not cal.is_working_day(x))
    # 中国节假日标记 - 添加日期验证
    def is_cn_holiday(date):
        try:
            return not calendar.is_workday(date)
        except NotImplementedError as e:
            # 打印出问题的日期以便调试
            print(f"日期超出范围警告: {date} - {str(e)}")
            # 默认按照周末判断
            return date.weekday() >= 5
    df['is_cn_holiday'] = df['Date'].apply(is_cn_holiday)
    # 节假日特征
    df['post_hk_holiday'] = df['is_hk_holiday'].shift(1).fillna(0).astype(int)
    df['post_cn_holiday'] = df['is_cn_holiday'].shift(1).fillna(0).astype(int)
    df['hk_weekend_holiday'] = df['is_weekend'] * df['is_hk_holiday']
    df['cn_weekend_holiday'] = df['is_weekend'] * df['is_cn_holiday']

    # 滞后特征
    for lag in [1, 2, 3, 4, 5, 7, 14, 21, 30]:
        df[f'lag_{lag}'] = df['Total'].shift(lag)
    for lag in [5, 7, 14]:
        df[f'ex_rate_lag_{lag}'] = df['ex'].shift(lag)
        for col in new_baidu_columns:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # 游客人数变化特征
    df['total_change_7d'] = df['lag_5'] - df['lag_7']
    df['total_change_14d'] = df['lag_7'] - df['lag_14']
    # 汇率变化特征
    df['ex_rate_change_14d'] = df['ex_rate_lag_7'] / df['ex_rate_lag_14'] - 1
    df['ex_rate_volatility_7d'] = df['ex_rate_lag_5'].shift().rolling(7).std()
    # 滚动统计量
    df['rolling_7_mean'] = df['Total'].shift().rolling(7).mean()
    df['rolling_7_std'] = df['Total'].shift().rolling(7).std()
    df['rolling_30_mean'] = df['Total'].shift().rolling(30).mean()
    df['ma_ratio_7_30'] = df['rolling_7_mean'] / df['rolling_30_mean']
    df['rolling_7_ex_rate_mean'] = df['ex_rate_lag_5'].shift().rolling(7).mean()

    # 傅里叶项（月度周期）
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)

    # 趋势特征
    df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days
    df['days_squared'] = df['days_since_start'] ** 2

    # 同比/环比
    df['weekly_growth'] = df['lag_7'] / df['lag_14']

    # 删除缺失值
    df.dropna(inplace=True)
    # 训练用数据
    train_df = df.copy()
    drop_columns = new_baidu_columns + ['ex']
    train_df.drop(drop_columns, axis=1, inplace=True)

    return df, train_df

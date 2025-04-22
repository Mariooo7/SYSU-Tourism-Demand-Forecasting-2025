# Hong Kong Tourism Flow Forecasting Project

## Project Introduction
This project is the final competition assignment for the 2024-2 semester course "Tourism Big Data Forecasting" at SYSBS. The objective is to forecast inbound passenger numbers at 5 major Hong Kong border crossings for the next 5 days, with daily submissions. It employs time series analysis models based on `statsforecast` and machine learning models based on `sklearn`.

## Main Project Structure
```
├── data/                # Raw data directory
│   └── round_/         # Data for each competition round
├── models/             # Trained models
│   └── round_/         # Models for each round
├── configs/            # Model parameters
│   ├── statsforecast_models.py  # Statsforecast parameters
│   ├── ml_models.py             # Sklearn parameters
├── src/                # Source code
│   ├── feature_engineering.py   # Feature engineering
│   ├── forecasting.py           # Time series training + prediction
│   ├── model_fit.py            # ML model training
│   └── model_predict.py        # ML model prediction
└── round_*.ipynb       # Analysis notebooks for each round
```

## External Data
1. `ex_rate.xlsx` in data directory stores HKD/RMB exchange rates (no cleaning required)
2. `data/baidu` stores daily Baidu Index data, cleaned in `clean_baidu.ipynb`
3. `data/event` stores major event data (generated and labeled by LLM), cleaned in `clean_event.ipynb` (subsequently removed due to poor performance)
4. `data/rf` stores rainfall data, cleaned in `clean_rf.ipynb`

## Exploratory Analysis
Performed in explore_*.ipynb notebooks to determine:
- Seasonality patterns
- Stationarity
- Cycle lengths
- Optimal steps for time series models

## Areas needing improvement:
Actual implementation didn't use multi-value step lists for time series model training/prediction, potentially causing errors
## Note: 
Saved models in models directory may be incomplete or contain redundant files

# 香港旅游流量预测项目

## 项目介绍
本项目为中山大学管理学院2024-2学期旅游大数据预测期末竞赛作业
目标为预测香港5个重要口岸的未来5日入境人数，每日提交一次
使用基于`statsforecast`的时间序列分析模型和基于`sklearn`的机器学习模型

## 主要项目结构
```
├── data/ # 原始数据目录 
│ └── round_/ # 各轮次数据 
├── models/ # 训练好的模型
│ └── round_/ # 各轮次模型
├── configs/ # 模型参数 
│ ├── statsforecast_models.py # statsforecast模型参数 
│ ├── ml_models.py # sklearn模型参数
├── src/ # 源代码 
│ ├── feature_engineering.py # 特征工程 
│ ├── forecasting.py # 时间序列模型训练+预测
│ ├── model_fit.py # 机器学习模型训练 
│ └── model_predict.py # 机器学习模型预测 
└── round_*.ipynb # 各轮次分析笔记本
```

## 外部数据
1. `data`目录下的`ex_rate.xlsx`存储港币兑人民币汇率，无需清洗
2. `date/baidu` 存储有关的日度百度指数，在`clean_baidu.ipynb`中清洗
3. `data/event` 存储有关的重大事件数据，由大语言模型生成并进行标注，在`clean_event.ipynb`中清洗(使用效果不佳，在后续预测中移除)
4. `data/rf` 存储降雨量数据，在`clean_rf.ipynb`中清洗

## 探索分析
`explore_*.ipynb`中进行对应口岸探索分析，确定季节性、是否平稳、周期长度、时间序列模型预测的最优步数

## 还需完善：
在实际使用中，未使用含有多个值的`step`列表进行时间序列模型的训练和预测，可能报错或不能正常训练
## 注意：
`models`目录下各轮次保存模型可能不全或有多余

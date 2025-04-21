# 香港旅游流量预测项目

## 项目介绍
本项目为中山大学管理学院2024-2学期旅游大数据预测期末竞赛作业
目标为预测香港5个重要口岸的未来5日入境人数，每日提交一次
使用基于statsforecast的时间序列分析模型和基于sklearn的机器学习模型

## 主要项目结构
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

## 外部数据
1. data目录下的ex_rate.xlsx存储港币兑人民币汇率，无需清洗
2. date/baidu 存储有关的日度百度指数，在clean_baidu.ipynb中清洗
3. data/event 存储有关的重大事件数据，在clean_event.ipynb中清洗
4. data/rf 存储降雨量数据，在clean_rf.ipynb中清洗

## 探索分析
explore_*.ipynb中进行对应口岸探索分析，确定季节性、是否平稳、周期长度、时间序列模型预测的最优步数

> 还需完善：在实际使用中，未使用含有多个值的step列表进行时间序列模型的训练和预测，可能报错或不能正常训练
>
> 注意：models目录下各轮次保存模型可能不全或有多余

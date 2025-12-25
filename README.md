# FDBD_exp4_Spark

## 项目简介
本项目基于 **Apache Spark** 平台，使用 **RDD、Spark SQL 和 Spark MLlib** 三种编程模型，对用户优惠券使用行为进行多维度分析。实验数据来源于 O2O 优惠券使用预测数据集，包含线上、线下消费记录。

## 实验环境
- **操作系统**：WSL2 (Ubuntu 20.04)
- **容器环境**：Docker
- **Spark 镜像**：`apache/spark:3.5.7-scala2.12-java11-python3-r-ubuntu`
- **Python 版本**：3.8.10
- **Spark 版本**：3.5.7

## 项目结构
/opt/spark/work-dir/
├── data/ # 数据集目录
│ ├── ccf_offline_stage1_train.csv
│ ├── ccf_online_stage1_train.csv
│ └── ccf_offline_stage1_test_revised.csv
├── task1.py # 任务一：Spark RDD 编程
├── task2.py # 任务二：Spark SQL 编程
├── task3.py # 任务三：Spark MLlib 编程
└── output/ # 输出结果目录

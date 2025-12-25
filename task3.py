from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import *
from pyspark.sql.types import *


# 1. 初始化Spark
spark = SparkSession.builder \
    .appName("Experiment4_Task3") \
    .master("local[*]") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("实验四任务三：Spark MLlib 优惠券使用预测")
print("=" * 60)

# 2. 数据加载与预处理
def load_and_preprocess_data():
    """加载并预处理数据"""
    
    # 读取离线训练数据
    train_df = spark.read.csv('/opt/spark/work-dir/data/ccf_offline_stage1_train.csv', 
                             header=True, 
                             inferSchema=True)
    
    # 读取测试数据
    test_df = spark.read.csv('/opt/spark/work-dir/data/ccf_offline_stage1_test_revised.csv', 
                            header=True, 
                            inferSchema=True)
    
    print(f"训练数据大小: {train_df.count()} 行")
    print(f"测试数据大小: {test_df.count()} 行")
    
    # 查看数据结构
    print("\n训练数据列名:", train_df.columns)
    print("\n测试数据列名:", test_df.columns)
    
    return train_df, test_df


def create_features(df, is_train=True):
    """创建特征"""
    
    # 复制数据框
    features_df = df
    
    # 1. 处理折扣率特征
    def parse_discount_rate(discount_rate):
        """解析折扣率，返回实际折扣率"""
        if discount_rate is None or discount_rate == 'null' or discount_rate == '':
            return None
        if ':' in discount_rate:
            try:
                parts = discount_rate.split(':')
                if len(parts) == 2:
                    # 折扣型，如 "0.9"
                    return float(parts[1])
                elif len(parts) == 3:
                    # 满减型，如 "100:10:00"
                    threshold = float(parts[0])
                    reduce = float(parts[1])
                    if threshold > 0:
                        return reduce / threshold  # 折扣率
            except:
                return None
        else:
            try:
                return float(discount_rate)
            except:
                return None
    
    # 注册UDF
    parse_discount_udf = udf(parse_discount_rate, DoubleType())
    
    # 2. 处理距离特征
    def parse_distance(distance):
        """解析距离，将null转为特定值"""
        if distance is None or distance == 'null' or distance == '':
            return 11.0  # 表示未知距离
        try:
            return float(distance)
        except:
            return 11.0
    
    parse_distance_udf = udf(parse_distance, DoubleType())
    
    # 3. 添加特征列
    features_df = features_df.withColumn("discount_rate_num", parse_discount_udf(col("Discount_rate")))
    features_df = features_df.withColumn("distance_num", parse_distance_udf(col("Distance")))
    
    # 4. 处理日期特征
    if is_train:
        # 训练数据：Date是使用日期，Date_received是领取日期
        # 创建标签：如果Date不为空且Date_received不为空，且使用日期在领取日期后15天内
        def create_label(date_received, date_used):
            if date_received is None or date_used is None:
                return 0
            if date_received == 'null' or date_used == 'null':
                return 0
            try:
                received_date = int(date_received)
                used_date = int(date_used)
                # 计算天数差
                days_diff = used_date - received_date
                # 如果在15天内使用，标记为1
                if 0 <= days_diff <= 15:
                    return 1
                else:
                    return 0
            except:
                return 0
        
        create_label_udf = udf(create_label, IntegerType())
        features_df = features_df.withColumn("label", create_label_udf(col("Date_received"), col("Date")))
    
    # 5. 时间特征 - 使用简单的方法避免复杂结构体
    def extract_month(date_str):
        if date_str is None or date_str == 'null' or len(str(date_str)) != 8:
            return None
        try:
            date_int = int(date_str)
            month = (date_int % 10000) // 100
            return month
        except:
            return None
    
    def extract_weekday(date_str):
        if date_str is None or date_str == 'null' or len(str(date_str)) != 8:
            return None
        try:
            date_int = int(date_str)
            day = date_int % 100
            weekday = (day % 7) + 1  # 简化的星期几计算
            return weekday
        except:
            return None
    
    def extract_is_weekend(date_str):
        if date_str is None or date_str == 'null' or len(str(date_str)) != 8:
            return 0
        try:
            date_int = int(date_str)
            day = date_int % 100
            weekday = (day % 7) + 1
            return 1 if weekday in [6, 7] else 0
        except:
            return 0
    
    def extract_is_end_of_year(date_str):
        if date_str is None or date_str == 'null' or len(str(date_str)) != 8:
            return 0
        try:
            date_int = int(date_str)
            month = (date_int % 10000) // 100
            return 1 if month in [11, 12] else 0
        except:
            return 0
    
    def add_coupon_type_features(features_df):
        """添加优惠券类型特征"""
        
        # 判断折扣类型
        def get_coupon_type(discount_rate):
            if discount_rate is None or discount_rate == 'null':
                return 0
            if ':' in discount_rate:
                parts = discount_rate.split(':')
                if len(parts) == 2:
                    return 1  # 折扣型
                elif len(parts) == 3:
                    return 2  # 满减型
            else:
                return 1  # 折扣型
        
        coupon_type_udf = udf(get_coupon_type, IntegerType())
        features_df = features_df.withColumn("coupon_type", coupon_type_udf(col("Discount_rate")))

    # 注册UDFs
    extract_month_udf = udf(extract_month, IntegerType())
    extract_weekday_udf = udf(extract_weekday, IntegerType())
    extract_is_weekend_udf = udf(extract_is_weekend, IntegerType())
    extract_is_end_of_year_udf = udf(extract_is_end_of_year, IntegerType())
    
    # 添加日期特征列
    features_df = features_df.withColumn("month", extract_month_udf(col("Date_received")))
    features_df = features_df.withColumn("weekday", extract_weekday_udf(col("Date_received")))
    features_df = features_df.withColumn("is_weekend", extract_is_weekend_udf(col("Date_received")))
    features_df = features_df.withColumn("is_end_of_year", extract_is_end_of_year_udf(col("Date_received")))

    features_df = add_coupon_type_features(features_df)
    

# 4. 主函数
def main():
    print("\n1. 加载数据...")
    train_df, test_df = load_and_preprocess_data()
    
    print("\n2. 创建特征...")
    train_features = create_features(train_df, is_train=True)
    test_features = create_features(test_df, is_train=False)
    
    print("\n训练数据特征创建完成，前5行:")
    train_features.select("User_id", "Merchant_id", "Coupon_id", "discount_rate_num", 
                         "distance_num", "month", "weekday", "is_weekend", "label").show(5)
    
    print("\n测试数据特征创建完成，前5行:")
    test_features.select("User_id", "Merchant_id", "Coupon_id", "discount_rate_num", 
                        "distance_num", "month", "weekday", "is_weekend").show(5)
    
    # 5. 准备特征向量
    print("\n3. 准备特征向量...")
    feature_cols = [
        "discount_rate_num",
        "distance_num",
        "month",
        "weekday",
        "is_weekend",
        "is_end_of_year"
    ]
    
    # 处理缺失值
    for col_name in feature_cols:
        train_features = train_features.fillna({col_name: 0})
        test_features = test_features.fillna({col_name: 0})
    
    # 创建特征向量
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    train_data = assembler.transform(train_features)
    test_data = assembler.transform(test_features)
    
    print("\n训练数据特征向量示例:")
    train_data.select("features", "label").show(5, truncate=False)
    
    # 6. 训练模型
    print("\n4. 训练逻辑回归模型...")
    
    # 划分训练集和验证集
    (train_set, val_set) = train_data.randomSplit([0.8, 0.2], seed=42)
    
    # 创建逻辑回归模型
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=10,
        regParam=0.01,
        elasticNetParam=0.5
    )
    
    # 训练模型
    model = lr.fit(train_set)
    
    print(f"模型训练完成，迭代次数: {model.summary.totalIterations}")
    
    # 7. 模型评估
    print("\n5. 模型评估...")
    
    # 在验证集上评估
    val_predictions = model.transform(val_set)
    
    # 计算准确率
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    
    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    auc = evaluator.evaluate(val_predictions)
    print(f"验证集AUC: {auc:.4f}")
    
    # 计算其他指标
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    
    accuracy = accuracy_evaluator.evaluate(val_predictions)
    print(f"验证集准确率: {accuracy:.4f}")
    
    # 8. 在测试集上预测
    print("\n6. 在测试集上预测...")
    test_predictions = model.transform(test_data)
    
    print("测试集预测结果示例:")
    test_predictions.select("User_id", "Merchant_id", "Coupon_id", 
                           "probability", "prediction").show(10, truncate=False)
    
    # 9. 保存预测结果
    print("\n7. 保存预测结果...")
    
    from pyspark.ml.functions import vector_to_array
    
    result_df = test_predictions.select(
        "User_id",
        "Merchant_id",
        "Coupon_id",
        "Date_received",
        vector_to_array(col("probability"))[1].alias("prob"),
        col("prediction").cast(IntegerType()).alias("prediction")
    )
    
    # 保存结果
    output_path = '/opt/spark/work-dir/output/task3_predictions'
    result_df.coalesce(1) \
        .write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(output_path)
    
    print(f"预测结果已保存到: {output_path}")
    
    # 10. 显示统计信息
    print("\n8. 预测结果统计:")
    total_count = test_predictions.count()
    positive_count = test_predictions.filter(col("prediction") == 1).count()
    positive_ratio = positive_count / total_count if total_count > 0 else 0
    
    print(f"总预测样本数: {total_count}")
    print(f"预测会使用优惠券的样本数: {positive_count}")
    print(f"预测使用率: {positive_ratio:.4f}")
    
    # 11. 可选：生成天池比赛提交格式
    print("\n9. 生成天池比赛提交格式...")
    
    # 定义UDF来提取正类概率（与保存预测结果时相同）
    def extract_probability(v):
        """从向量中提取正类概率"""
        if v is None:
            return 0.0
        try:
            # 向量转换为数组，取第二个元素（正类概率）
            return float(v.toArray()[1])
        except:
            return 0.0
    
    extract_prob_udf = udf(extract_probability, DoubleType())
    
    # 天池提交格式：User_id,Coupon_id,Date_received,Probability
    submission_df = test_predictions.select(
        "User_id",
        "Coupon_id",
        "Date_received",
        extract_prob_udf(col("probability")).alias("Probability")
    )
    
    submission_path = '/opt/spark/work-dir/output/tianchi_submission.csv'
    submission_df.coalesce(1) \
        .write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(submission_path)
    
    print(f"天池提交文件已保存到: {submission_path}")
    
    print("任务三完成！")
    
    return model, test_predictions

if __name__ == "__main__":
    try:
        model, predictions = main()
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()
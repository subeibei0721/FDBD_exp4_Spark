from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# 1. 初始化SparkSession
spark = SparkSession.builder \
    .appName("Experiment4_Task2") \
    .master("local[*]") \
    .getOrCreate()

print("\n1. 优惠券使用时间分布统计")
offline_df = spark.read.csv('/opt/spark/work-dir/data/ccf_offline_stage1_train.csv', 
                           header=True, 
                           inferSchema=True)

offline_df.createOrReplaceTempView("offline_data")

# SQL查询：统计每个优惠券上中下旬使用概率
coupon_time_distribution_sql = """
SELECT 
    Coupon_id,
    ROUND(SUM(CASE WHEN day_of_month <= 10 THEN 1 ELSE 0 END) / COUNT(*), 4) AS early_prob,
    ROUND(SUM(CASE WHEN day_of_month > 10 AND day_of_month <= 20 THEN 1 ELSE 0 END) / COUNT(*), 4) AS mid_prob,
    ROUND(SUM(CASE WHEN day_of_month > 20 THEN 1 ELSE 0 END) / COUNT(*), 4) AS late_prob
FROM (
    SELECT 
        Coupon_id,
        CAST(SUBSTRING(Date, 7, 2) AS INT) AS day_of_month
    FROM offline_data
    WHERE Date IS NOT NULL
        AND Date != 'null'
        AND Date != '' 
        AND Coupon_id IS NOT NULL 
        AND Coupon_id != 'null'
        AND LENGTH(Date) = 8
) valid_data
GROUP BY Coupon_id
ORDER BY Coupon_id
"""

coupon_time_distribution = spark.sql(coupon_time_distribution_sql)

# 显示前10行结果
print("优惠券使用时间分布（前10行）：")
coupon_time_distribution.show(10)

# 保存结果
coupon_time_distribution.write \
    .mode("overwrite") \
    .csv('/opt/spark/work-dir/output/task2_coupon_time_distribution')

print("\n2. 商家正样本比例统计")

online_data_path = '/opt/spark/work-dir/data/task1_2_online_consumption_table.csv'

schema = StructType([
    StructField("Merchant_id", StringType(), True),
    StructField("negative_count", IntegerType(), True),
    StructField("normal_count", IntegerType(), True),
    StructField("positive_count", IntegerType(), True)
])

online_df = spark.read.csv(online_data_path, header=True, schema=schema)
online_df.createOrReplaceTempView("online_consumption_table")

merchant_positive_ratio_sql = """
SELECT
    Merchant_id,
    ROUND(positive_count * 1.0 / (negative_count + normal_count + positive_count), 4) AS positive_ratio,
    positive_count,
    (negative_count + normal_count + positive_count) AS total_count
FROM online_consumption_table
WHERE (negative_count + normal_count + positive_count) > 0
ORDER BY positive_ratio DESC
LIMIT 10
"""

top_merchants = spark.sql(merchant_positive_ratio_sql)
# 显示结果
print("正样本比例最高的前10个商家：")
top_merchants.show(10, truncate=False)

# 保存结果
top_merchants.write \
    .mode("overwrite") \
    .csv('/opt/spark/work-dir/output/top_merchants_positive_ratio')
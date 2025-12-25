from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *

# 初始化Spark
conf = SparkConf().setAppName("Experiment4_Task1").setMaster("local[*]")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# 线上数据文件的schema（根据实际文件）
def get_online_schema():
    return StructType([
        StructField("User_id", StringType(), True),
        StructField("Merchant_id", StringType(), True),
        StructField("Action", StringType(), True),
        StructField("Coupon_id", StringType(), True),
        StructField("Discount_rate", StringType(), True),
        StructField("Date_received", StringType(), True),
        StructField("Date", StringType(), True)
    ])

def task1_coupon_usage_count():
    online_data_path = '/opt/spark/work-dir/data/ccf_online_stage1_train.csv'
    
    print(f"读取文件: {online_data_path}")
    
    schema = get_online_schema()
    online_df = spark.read.csv(online_data_path, header=True, schema=schema)
      
    online_rdd = online_df.rdd

    # 正样本条件
    positive_sample = online_rdd.filter(lambda row: 
        row['Date'] is not None and
        row['Date'] != 'null' and
        row['Date'] != '' and
        row['Coupon_id'] is not None and
        row['Coupon_id'] != 'null' and
        row['Coupon_id'] != '' and
        row['Coupon_id'] != 'fixed'
    )
    
    print(f"\n正样本数量: {positive_sample.count()}")
    
    coupon_counts_rdd = positive_sample.map(lambda row: (row['Coupon_id'], 1))
    
    coupon_counts = coupon_counts_rdd.reduceByKey(lambda a, b: a + b)
    sorted_coupon_counts = coupon_counts.sortBy(lambda x: x[1], ascending=False)
    
    top_10_coupons = sorted_coupon_counts.take(10)
    
    print("\n任务一：优惠券使用次数统计（前10名）")
    print("Coupon_id\t总使用次数")
    for coupon_id, count in top_10_coupons:
        print(f"{coupon_id}\t{count}")
    
    # 保存结果
    output_path = '/opt/spark/work-dir/output/task1_coupon_counts'
    sorted_coupon_counts.map(lambda x: f"{x[0]},{x[1]}").saveAsTextFile(output_path)
    print(f"\n结果已保存到: {output_path}")
    
    return sorted_coupon_counts

def task2_merchant_coupon_analysis():
    online_data_path = '/opt/spark/work-dir/data/ccf_online_stage1_train.csv'
    
    print(f"读取文件: {online_data_path}")
    
    # 使用相同的schema
    schema = get_online_schema()
    online_df = spark.read.csv(online_data_path, header=True, schema=schema)     
    online_rdd = online_df.rdd

    def label_record(row):
        merchant_id = row['Merchant_id']
        date = row['Date']
        coupon_id = row['Coupon_id']

        # 判断是否为null或空字符串
        date_is_null = (date is None or date == 'null' or date == '')
        coupon_is_null = (coupon_id is None or coupon_id == 'null' or coupon_id == '' or coupon_id == 'fixed')

        if date_is_null and not coupon_is_null:
            label = 'negative'
        elif not date_is_null and coupon_is_null:
            label = 'normal'
        elif not date_is_null and not coupon_is_null:
            label = 'positive'
        else:
            label = 'other'

        return (merchant_id, (label, 1))
    
    labeled_rdd = online_rdd.map(label_record)

    # 过滤掉'other'类型
    filtered_rdd = labeled_rdd.filter(lambda x: x[1][0] != 'other')
    
    # 继续原有处理
    mapped_rdd = filtered_rdd.map(lambda x: ((x[0], x[1][0]), x[1][1]))
    count_by_merchant_label = mapped_rdd.reduceByKey(lambda a, b: a + b)

    def reorganize_data(item):
        (merchant_id, label), count = item

        negative = 0
        normal = 0
        positive = 0

        if label == 'negative':
            negative = count
        elif label == 'normal':
            normal = count
        elif label == 'positive':
            positive = count

        return (merchant_id, (negative, normal, positive))
    
    reorganized_rdd = count_by_merchant_label.map(reorganize_data)

    def merge_counts(counts1, counts2):
        n1, no1, p1 = counts1
        n2, no2, p2 = counts2
        return (n1 + n2, no1 + no2, p1 + p2)
    
    merchant_counts = reorganized_rdd.reduceByKey(merge_counts)
    sorted_merchant_counts = merchant_counts.sortBy(lambda x: x[0])
    
    top10_merchants = sorted_merchant_counts.take(10)

    # 转换为DataFrame
    result_rows = sorted_merchant_counts.map(lambda x: 
        (x[0], x[1][0], x[1][1], x[1][2]))
    
    result_schema = StructType([
        StructField("Merchant_id", StringType(), True),
        StructField("negative_count", IntegerType(), True),
        StructField("normal_count", IntegerType(), True),
        StructField("positive_count", IntegerType(), True)
    ])

    result_df = spark.createDataFrame(result_rows, schema=result_schema)

    # 注册为临时表
    result_df.createOrReplaceTempView("online_consumption_table")

    # 保存到文件
    output_path = '/opt/spark/work-dir/output/task1_online_consumption_table'
    result_df.write.mode("overwrite").csv(output_path)

    print("\n任务二：商家优惠券使用情况（前10行）")
    print("Merchant_id\t负样本数量\t普通消费数量\t正样本数量")
    for merchant_id, counts in top10_merchants:
        negative, normal, positive = counts
        print(f"{merchant_id}\t{negative}\t{normal}\t{positive}")
    
    print(f"\n结果已保存到: {output_path}")
    
    return result_df

def main():
    
    # 创建输出目录
    import os
    os.makedirs('/opt/spark/work-dir/output', exist_ok=True)
    
    # 执行任务一
    print("\n执行任务一：统计优惠券发放数量...")
    coupon_counts = task1_coupon_usage_count()
    
    # 执行任务二
    print("\n执行任务二：查询指定商家优惠券使用情况...")
    merchant_analysis_df = task2_merchant_coupon_analysis()
       
    print("\n实验任务一完成！")
    
    return coupon_counts, merchant_analysis_df

if __name__ == "__main__":
    coupon_results, merchant_results = main()
    sc.stop()
    print("\nSpark已停止")
'''
此脚本包含了创建数据库表 插入数据和查询所有记录的操作 旨在处理样本数据

创建数据库表

    函数: create_sqlite()
    功能: 创建一个名为SampleData的表 用于存储样本信息
    表结构:
    SampleID: 自增的样本ID 主键
    Barcode: 样本的条形码
    TubeType: 管子的类型
    BloodLayerInfo: 血液层信息
    BloodHeight: 血液高度
    CollectionBatch: 采集批次
    CollectionTime: 采集时间

插入数据

    函数: insert_sample_data(barcode, tube_type, blood_layer_info, blood_height, collection_batch, collection_time)
    功能: 插入一条新的样本数据记录。
    参数:
    barcode: 样本的条形码
    tube_type: 管子的类型
    blood_layer_info: 血液层信息
    blood_height: 血液高度
    collection_batch: 采集批次
    collection_time: 采集时间
    
查询所有记录

    函数: fetch_all_samples()
    功能: 查询并打印所有样本数据记录
'''

import sqlite3
from datetime import datetime

def create_sqlite():
    # 连接到SQLite数据库
    # 如果数据库不存在 则会自动创建
    conn = sqlite3.connect('./src/database/sample_data.db')

    # 创建一个游标对象
    cursor = conn.cursor()
    # 创建SampleData表的SQL语句
    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS SampleData (
        SampleID INTEGER PRIMARY KEY AUTOINCREMENT,
        Barcode TEXT,
        TubeType TEXT,
        BloodLayerInfo TEXT,
        BloodHeight TEXT,
        CollectionBatch INT,
        CollectionTime DATETIME
    );
    '''

    cursor.execute(create_table_sql)
    conn.commit()
    cursor.close()
    conn.close()

    print("SampleData表创建成功。")

def insert_sample_data(barcode, tube_type, blood_layer_info, blood_height, collection_batch, collection_time):
    # 连接到SQLite数据库
    conn = sqlite3.connect('./src/database/sample_data.db')
    cursor = conn.cursor()
    
    # 插入数据的SQL语句
    insert_sql = '''
    INSERT INTO SampleData (Barcode, TubeType, BloodLayerInfo, BloodHeight, CollectionBatch, CollectionTime)
    VALUES (?, ?, ?, ?, ?, ?)
    '''
    
    cursor.execute(insert_sql, (barcode, tube_type, blood_layer_info, blood_height, collection_batch, collection_time))
    conn.commit()
    cursor.close()
    conn.close()

    print("数据插入成功。")

def fetch_all_samples():
    conn = sqlite3.connect('./src/database/sample_data.db')
    cursor = conn.cursor()
    
    # 查询所有记录的SQL语句
    fetch_all_sql = 'SELECT * FROM SampleData'
    cursor.execute(fetch_all_sql)
    records = cursor.fetchall()

    for record in records:
        print(record)
    
    cursor.close()
    conn.close()
    return records

if __name__ == '__main__':
    # 创建数据库表
    create_sqlite()
    
    # 插入样本数据
    insert_sample_data('BC123456', 'TypeA', 'Layer1', '10cm', 1, datetime.now())
    insert_sample_data('BC123457', 'TypeB', 'Layer2', '12cm', 1, datetime.now())
    insert_sample_data('BC123458', 'TypeC', 'Layer3', '15cm', 1, datetime.now())

    # 查询并打印所有样本数据
    fetch_all_samples()
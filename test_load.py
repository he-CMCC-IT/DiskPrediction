# -*- coding: UTF-8 -*-
import csv
import os
import re
import pymysql

'''
将1天前?25个字段，
一共7万多条数据导入到mysql，
几秒钟完成
'''

# pymysql.install_as_MySQLdb()
# import MySQLdb

db = pymysql.connect("localhost", "root", "8888888", "test", charset='utf8')
cursor = db.cursor()
cursor.execute("SELECT VERSION()")
data = cursor.fetchone()
print("Database version : %s " % data)

cursor = db.cursor()
cursor.execute("drop table if exists disk_0")
sql = """create table disk_0 (
         date_time varchar(50),
         serial_number varchar(50) not null,
         model varchar(50),
         capacity_bytes varchar(50),
         failure varchar(50),
         smart_1_normalized varchar(50),
         smart_1_raw varchar(50),
         smart_2_normalized varchar(50),
         smart_2_raw varchar(50),
         smart_3_normalized varchar(50),
         smart_3_raw varchar(50),
         smart_4_normalized varchar(50),
         smart_4_raw varchar(50),
         smart_5_normalized varchar(50),
         smart_5_raw varchar(50),
         smart_6_normalized varchar(50),
         smart_6_raw varchar(50),
         smart_7_normalized varchar(50),
         smart_7_raw varchar(50),
         smart_8_normalized varchar(50),
         smart_8_raw varchar(50), 
         smart_9_normalized varchar(50),
         smart_9_raw varchar(50),
         smart_10_normalized varchar(50),
         smart_10_raw varchar(50),
         primary key(serial_number)
)charset = utf8mb4;
"""
cursor.execute(sql)


# from sqlalchemy import create_engine
# sqlcont = create_engine('mysql://root:root@127.0.0.1:3306/test?charset=utf8')

def traverse(f):
    expr = re.compile("\d{4}-\d{2}-\d{2}.csv")
    fs = os.listdir(f)
    file_num = 0
    for f1 in fs:
        tmp_path = os.path.join(f, f1)
        if not os.path.isdir(tmp_path):
            # print('文件: %s' % tmp_path)
            if re.search(expr, tmp_path):
                print('文件: %s' % tmp_path)
                data = csv.reader(open(tmp_path, 'r'))
                head = next(data)
                sql = 'insert into disk_0 values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
                for item in data:
                    cursor.execute(sql, tuple(item[0:25]))
        else:
            traverse(tmp_path)
        file_num += 1
        if file_num == 1:
            break



path = r'D:\a学习和工作\data_Q1_2017'
traverse(path)
db.commit()

cursor.close()
db.close()

# 批量insert
# for t in range(0,100):
#
#      sql = 'insert into tb5 (id, val) values '
#
#      for i in range(1,100000):
#             sql += ' ('+`t*100000+i`+', "tb5EXTRA"),'
#      sql += ' ('+`t`+'00000, "tb5EXTRA")'
#
#      cur.execute(sql)
#
#      db.commit()
#
#  cur.close()
#
#  db.close()

# -*- coding: UTF-8 -*-
import csv
import os
import re
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb

db = MySQLdb.connect("localhost", "root", "123123", "test", charset='utf8' )
cursor = db.cursor()
cursor.execute("SELECT VERSION()")
data = cursor.fetchone()
print("Database version : %s " % data)

# from sqlalchemy import create_engine
# sqlcont = create_engine('mysql://root:root@127.0.0.1:3306/test?charset=utf8')

def traverse(f):
    expr = re.compile("\d{4}-\d{2}-\d{2}.csv")
    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f, f1)
        if not os.path.isdir(tmp_path):
            #print('文件: %s' % tmp_path)
            if re.search(expr,tmp_path):
                print('文件: %s' % tmp_path)
                tmpfile = csv.reader(open(tmp_path,'r'))
                target = 0
                for line in tmpfile:
                    print(line)
                    target+=1
                    if target >=10:
                        break
        else:
            #print('文件夹：%s' % tmp_path)
            traverse(tmp_path)


path = r'D:\baidudisk\data_Q1_2017'
traverse(path)
db.close()

#批量insert
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
import os
import pickle
import numpy as np
import schedule
import time
import datetime
import shutil
import configparser
import tkinter
import tkinter.messagebox

'''

每隔一天检测一次所有机器的硬盘运行状态
采集的数据也是天更新一次，按本程序设定，需要在本程序执行的1天内采集得到第一次（也就是第一天）的数据

'./data/'中放着历史磁盘数据（格式是'IP地址_时间.txt'）
'./data2/'中放着采集的当天的数据

步骤1：加载rf模型
步骤2：从txt文件中提取属性数据，并做标准化（步骤2还包括补充没有采集到的属性数据）
步骤3：将上一步骤得到的特征放入到模型中，进行预测，得到结果
步骤4：将当前文件夹的数据移到历史数据文件夹

'''


def MaxMinNormalization(x, Max, Min):
    if x < Min:
        x = Min
    elif x > Max:
        x = Max
    x_normal = 2 * ((x - Min) / (Max - Min)) - 1
    return x_normal


# 从配置文件中读取rf模型路径
conf = configparser.ConfigParser()
conf.read('conf.ini')

path_model = conf.get('config', 'path_model')
path_data_yidong_current = conf.get('config', 'path_data_yidong_current')
path_data_yidong_history = conf.get('config', 'path_data_yidong_history')

# 加载rf.pickle
with open(path_model + 'rf.pickle', 'rb') as fr:
    model = pickle.load(fr)


def move_files(dir1, dir2):
    for filename in os.listdir(dir1):
        shutil.move(os.path.join(dir1, filename), dir2)


def job():
    print("Start: %s" % datetime.datetime.now())

    files = os.listdir(path_data_yidong_current)
    if files is not None:
        tkinter.messagebox.showwarning('警告', '文件夹是空的！')

    data = []
    disk_ids = []

    for filename in files:
        sub = []

        id = filename.split("_")[0]
        disk_ids.append(id)

        with open(path_data_yidong_current + filename, "r") as f:
            for line in f:
                if "Raw_Read_Error_Rate" in line:
                    value = int(line.split()[3])
                    value = MaxMinNormalization(value, 32, 200)
                    # value = round(value, 2)
                    sub.append(value)
                elif "Spin_Up_Time" in line:
                    value = int(line.split()[3])
                    value = MaxMinNormalization(value, 83, 253)
                    # value = round(value, 2)
                    sub.append(value)
                elif "Reallocated_Sector_Ct" in line:
                    value = int(line.split()[3])
                    value = MaxMinNormalization(value, 5, 252)
                    # value = round(value, 2)
                    sub.append(value)
                    raw_value = int(line.split()[-1])
                    raw_value = MaxMinNormalization(raw_value, 0, 43248)
                elif "Seek_Error_Rate" in line:
                    value = int(line.split()[3])
                    value = MaxMinNormalization(value, 38, 252)
                    # value = round(value, 2)
                    sub.append(value)
                elif "Power_On_Hours" in line:
                    value = int(line.split()[3])
                    value = MaxMinNormalization(value, 11, 100)
                    # value = round(value, 2)
                    sub.append(value)
                elif "Temperature_Celsius" in line:
                    value = int(line.split()[3])
                    value = MaxMinNormalization(value, 12, 253)
                    # value = round(value, 2)
                    sub.append(value)
                else:
                    continue
            sub.append(raw_value)
        data.append(sub)

    data = np.array(data)
    result = model.predict(data)

    if sum(result) == 0:
        for id in disk_ids:
            print("%s硬盘状态良好！" % id)
    elif sum(result) > 0:
        indexes = np.where(result == 1)[0]
        for i in indexes:
            print("%s有硬盘将在10天内故障！" % (disk_ids[i]))

    print("End: %s" % datetime.datetime.now())

    move_files(path_data_yidong_current, path_data_yidong_history)


schedule.every().day.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)

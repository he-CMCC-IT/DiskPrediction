import os
import pickle
import numpy as np
import schedule
import time
import datetime
import shutil
import configparser

from RedisPublisher import RedisPublisher

'''

每隔一小时检测一次所有磁盘的运行状态
采集的数据也是每小时更新一次，按本程序设定，需要在本程序执行的1小时内采集得到第一次（也就是第一小时）的数据

'./data/'中放着历史磁盘数据（格式是'磁盘id_时间.txt'的吸收那个会）
'./data2/'中放着采集的当前小时的数据

步骤1：加载svm模型
步骤2：从txt文件中提取属性数据，并做标准化（步骤2还包括补充没有采集到的属性数据）
步骤3：将上一步骤得到的特征放入到模型中，进行预测，得到结果
步骤4：将当前文件夹的数据移到历史数据文件夹

'''


def MaxMinNormalization(x, Max, Min):
    x_normal = 2 * ((x - Min) / (Max - Min)) - 1
    return x_normal


# 从配置文件中读取svm模型路径，移动公司自己的磁盘当前运行数据文件路径和历史数据文件路径
conf = configparser.ConfigParser()
conf.read('conf.ini')

path_model = conf.get('config', 'path_model')
path_data_yidong_current = conf.get('config', 'path_data_yidong_current')
path_data_yidong_history = conf.get('config', 'path_data_yidong_history')

# 加载svm.pickle, 这个保存的模型在测试数据上达到了75.19%的检出率，
# 0.30%的误检率，可以平均提前355.66小时进行预测
with open(path_model + 'svm.pickle', 'rb') as fr:
    clf = pickle.load(fr)


def move_files(dir1, dir2):
    for filename in os.listdir(dir1):
        shutil.move(os.path.join(dir1, filename), dir2)


def job():
    print("Start: %s" % datetime.datetime.now())
    files = os.listdir(path_data_yidong_current)
    # pred_files = [filename for filename in files if datetime.datetime.now().hour == int(filename[-8:-6])]
    # print(filename)
    data = []
    disk_ids = []

    # for filename in pred_files:
    for filename in files:
        sub = []

        id = filename.split("_")[0]
        disk_ids.append(id)

        with open(path_data_yidong_current + filename, "r") as f:
            for line in f:
                if "Raw_Read_Error_Rate" in line:
                    value = int(line.split()[3])
                    value = MaxMinNormalization(value, 1, 253)
                    # value = round(value, 2)
                    sub.append(value)
                elif "Spin_Up_Time" in line:
                    value = int(line.split()[3])
                    value = MaxMinNormalization(value, 1, 253)
                    # value = round(value, 2)
                    sub.append(value)
                elif "Reallocated_Sector_Ct" in line:
                    value = int(line.split()[3])
                    value = MaxMinNormalization(value, 1, 253)
                    # value = round(value, 2)
                    sub.append(value)
                    raw_value = int(line.split()[-1])
                    raw_value = MaxMinNormalization(raw_value, 0, 43248)
                elif "Seek_Error_Rate" in line:
                    value = int(line.split()[3])
                    value = MaxMinNormalization(value, 1, 253)
                    # value = round(value, 2)
                    sub.append(value)
                elif "Power_On_Hours" in line:
                    value = int(line.split()[3])
                    value = MaxMinNormalization(value, 1, 253)
                    # value = round(value, 2)
                    sub.append(value)
                    # 添加187、189的属性值
                    value = MaxMinNormalization(100, 1, 253)
                    sub.append(value)
                    sub.append(value)
                elif "Temperature_Celsius" in line:
                    value = int(line.split()[3])
                    value = MaxMinNormalization(value, 1, 253)
                    # value = round(value, 2)
                    sub.append(value)
                    # 添加195、197的属性值
                    value = MaxMinNormalization(22, 1, 253)
                    sub.append(value)
                    value = MaxMinNormalization(100, 1, 253)
                    sub.append(value)
                else:
                    continue
            sub.append(raw_value)
            # 添加197属性的原始值
            raw_value = MaxMinNormalization(0, 0, 2432)
            sub.append(raw_value)
        data.append(sub)

    data = np.array(data)
    result = clf.predict(data)
    rd = RedisPublisher()
    if sum(result) == 0:
        rd.pubMsg("mychannel","所有磁盘都处在健康状态")
        print('所有磁盘都处在健康状态')
    elif sum(result) > 0:
        indexs = np.where(result == 1)[0]
        for i in indexs:
            print("磁盘" + disk_ids[i] + "大约在355个小时候发生故障！")
            rd.pubMsg("mychannel", "磁盘" + disk_ids[i] + "大约在355个小时候发生故障！")


    print("End: %s" % datetime.datetime.now())
    # print(disk_ids)

    move_files(path_data_yidong_current, path_data_yidong_history)


# schedule.every().hour.do(job)
#
# while True:
#     schedule.run_pending()
#     time.sleep(1)

if __name__=="__main__":
    job()
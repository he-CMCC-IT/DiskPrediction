# 从文档解析数据，并对数据进行归一化
# 此程序使用7个属性
import os
import pickle
import numpy as np
import pandas as pd

def MaxMinNormalization(x, Max, Min):
    x_normal = 2*((x - Min) / (Max - Min)) - 1
    return x_normal

data = []
disk_ids = []
i_disk = -1
for filename in os.listdir("./data/"):
    print(filename)
    sub = []
    disk_id = filename.split("_")[0]
    if disk_id not in disk_ids:
        disk_ids.append(disk_id)
        i_disk += 1
    sub.append(i_disk)
    with open("./data/" + filename, "r") as f:
        for line in f:
            if "Raw_Read_Error_Rate" in line:
                value = int(line.split()[3])
                value = MaxMinNormalization(value, 1, 253)
                value = round(value, 2)
                sub.append(value)
            elif "Spin_Up_Time" in line:
                value = int(line.split()[3])
                value = MaxMinNormalization(value, 1, 253)
                value = round(value, 2)
                sub.append(value)
            elif "Reallocated_Sector_Ct" in line:
                value = int(line.split()[3])
                value = MaxMinNormalization(value, 1, 253)
                value = round(value, 2)
                sub.append(value)
                raw_value = int(line.split()[-1])
                raw_value = MaxMinNormalization(raw_value, 0, 43248)
            elif "Seek_Error_Rate" in line:
                value = int(line.split()[3])
                value = MaxMinNormalization(value, 1, 253)
                value = round(value, 2)
                sub.append(value)
            elif "Power_On_Hours" in line:
                value = int(line.split()[3])
                value = MaxMinNormalization(value, 1, 253)
                value = round(value, 2)
                sub.append(value)
            elif "Temperature_Celsius" in line:
                value = int(line.split()[3])
                value = MaxMinNormalization(value, 1, 253)
                value = round(value, 2)
                sub.append(value)
            else:
                continue
        sub.append(raw_value)
    data.append(sub)
print(data)
# print(disk_ids)




# 加载svm.pickle，根据数据预测结果

data = np.array(data)
disk_index = data[:, 0].round().astype('int32')
with open('./svm.pickle', 'rb') as fr:
    clf = pickle.load(fr)
    print(clf.predict(data[:, 1:]))
print(disk_index)
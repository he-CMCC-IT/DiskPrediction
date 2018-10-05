import numpy as np
import sklearn
from sklearn import svm
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from sklearn.ensemble import RandomForestClassifier
import configparser


def getTrainTestIndex(serial_number, scale=0.3):
    '''
    get train and test number for good and bad drive respectively
    bad drive: start=1, end=433
    good drive: start=434, end=23395
    '''
    train_serial = serial_number
    test_serial = []
    test_num = int(len(train_serial) * scale)
    for i in range(test_num):
        # np.random.seed(i)
        randomIndex = int(np.random.uniform(0, len(train_serial)))
        test_serial.append(train_serial[randomIndex])
        del train_serial[randomIndex]
    return train_serial, test_serial


# 读取配置文件中百度数据文件路径和svm模型保存路径
conf = configparser.ConfigParser()
conf.read('conf_backblaze.ini')

path_ST4000DM000_data_one_csv =conf.get('config', 'path_ST4000DM000_data_one_csv')
path_model_backblaze = conf.get('config', 'path_model_backblaze')

# read data
# 百度数据500多兆，暂未上传
file_path = path_ST4000DM000_data_one_csv + 'ST4000DM000_sort_normalized.csv'
data = pd.read_csv(open(file_path))
print(data.shape)
# print(data[0, :])

# data_new = np.hstack((data[:, 0:7], data[:, 9].reshape(-1, 1), data[:, 12].reshape(-1, 1)))
# print(data_new.shape)
# print(data_new[0, :])

print('数据加载完成')

# get train and test data
data_bad = data.iloc[:12080, :]
data_good = data.iloc[12080:, :]
del data
# data_bad = data_new[:156312, :]
# data_good = data_new[156312:, :]

# get train and test serial number for good and bad drive respectively
serial_bad = list(set(data_bad['serial_number']))
serial_good = list(set(data_good['serial_number']))
train_token_bad, test_token_bad = getTrainTestIndex(serial_bad)
train_token_good, test_token_good = getTrainTestIndex(serial_good)

train_data_bad = data_bad[data_bad['serial_number'].isin(train_token_bad)]
test_data_bad = data_bad[data_bad['serial_number'].isin(test_token_bad)]
del data_bad

train_data_good = data_good[data_good['serial_number'].isin(train_token_good)]
test_data_good = data_good[data_good['serial_number'].isin(test_token_good)]
del data_good

# get data for train of bad
training_bad = train_data_bad.groupby('serial_number').tail(n=12)
del train_data_bad

# get data for train of good
training_good = train_data_good.groupby('serial_number').apply(lambda df: df.sample(n=4))
del train_data_good

# get training data of x and y
training_bad_x = np.array(training_bad.iloc[:, 3:])
training_bad_y = np.array([1] * len(training_bad), dtype='int32')
del training_bad

training_good_x = np.array(training_good.iloc[:, 3:])
training_good_y = np.array([0] * len(training_good), dtype='int32')
del training_good

training_x = np.vstack((training_bad_x, training_good_x))
training_y = np.concatenate((training_bad_y, training_good_y))

# get random training data
index = [i for i in range(len(training_x))]
# random.seed(5)
random.shuffle(index)
training_x_ = training_x[index]
training_y_ = training_y[index]
print('训练数据准备完毕')

# model
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(training_x_, training_y_)
# model = RandomForestClassifier(n_estimators=200, oob_score=True, min_samples_leaf=5, max_depth=10)
# model.fit(training_x_, training_y_)
# model.fit(training_x, training_y)
print('模型训练完成')

# 保存成python支持的文件格式pickle, 在当前目录下可以看到svm.pickle
with open(path_model_backblaze + 'svm.pickle', 'wb') as fw:
    pickle.dump(clf, fw)

# test_data_bad, test_data_good  *Evaluation!Evaluation!*
test_bad_x = np.array(test_data_bad.iloc[:, 3:])
test_bad_y = np.ones((len(test_data_bad),), dtype='int32')
test_bad_index = test_data_bad['serial_number']
del test_data_bad

test_good_x = np.array(test_data_good.iloc[:, 3:])
test_good_y = np.zeros((len(test_data_good),), dtype='int32')
test_good_index = test_data_good['serial_number']
del test_data_good

print('测试数据准备完成')

y_bad_pre = clf.predict(test_bad_x)
y_good_pre = clf.predict(test_good_x)
# y_bad_pre = model.predict(test_bad_x)
# y_good_pre = model.predict(test_good_x)
print('预测完成')

# Metric: (y_bad_pre, test_bad_index) (y_good_pre, test_good_index)
num_bad_true = len(test_token_bad)
num_good_true = len(test_token_good)

a, place_single_bad = np.unique(test_bad_index, return_index=True)
a, place_single_good = np.unique(test_good_index, return_index=True)

# num of predicting bad correctly
num_bad_pre = 0  # 96: 125/129=96.90%; 48: 112 FDR=112/129=86.82%; 24: 103/129=79.84%; 12: 99/129=76.74%
time_pre = []  # 96: 352.90; 48: avg=348.54; 24: 356.55; 12: 357.22
place_single_bad = list(place_single_bad)
place_single_bad.append(len(test_bad_index))
for i in range(len(place_single_bad) - 1):
    sub_pre = y_bad_pre[place_single_bad[i]:place_single_bad[i + 1]]
    if i == 1:
        print(test_bad_index[place_single_bad[i]-1:place_single_bad[i + 1]+1])
    if sum(sub_pre) > 0:
        num_bad_pre += 1
        time_pre.append(len(sub_pre) - np.where(sub_pre == 1)[0][0] - 1)

# 检出率
FDR = num_bad_pre / num_bad_true
print(FDR)
# 平均预测提前时间
print(sum(time_pre) / num_bad_pre)

# num of predicting good to bad
num_good_pre = 0
place_single_good = list(place_single_good)
place_single_good.append(len(test_good_index))
for i in range(len(place_single_good) - 1):
    sub_pre = y_good_pre[place_single_good[i]:place_single_good[i + 1]]
    if sum(sub_pre) > 0:
        num_good_pre += 1

# 误检率
FAR = num_good_pre / num_good_true
print(FAR)  # 96: 110/6888=1.60% ;48: 77/6888=1.12%; 24: 52/6888=0.75%; 12: 28/6888=0.41%
print('评价指标得到')

mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.hist(time_pre, 10, (0, 500), color='Blue')
plt.xticks((0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500))
plt.xlabel('提前的小时数')
plt.ylabel('被正确预测的坏盘数')
plt.show()
print('提前小时数分布直方图')

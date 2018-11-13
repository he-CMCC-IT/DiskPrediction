import numpy as np
import sklearn
from sklearn import svm
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import configparser
import functions

# read data
files_path = 'D:/a学习和工作/data_Q3_2017/'
file_path = files_path + 'ST4000DM000_sort_sample_time_series_kDifferent_alpha07_normalized.csv'
# data = pd.read_csv(open(file_path), index_col=0)
data = pd.read_csv(open(file_path))
print('数据大小：', data.shape)
print('数据加载完成')

# get train and test data
data_bad, data_good = functions.split_bad_good_disks(data)
del data

serial_bad = list(set(data_bad['serial_number']))
serial_good = list(set(data_good['serial_number']))
# serial_good = serial_good[:1000]

train_serial_bad, test_serial_bad = functions.get_train_test_index(serial_bad)
train_serial_good, test_serial_good = functions.get_train_test_index(serial_good)

train_data_bad = data_bad[data_bad['serial_number'].isin(train_serial_bad)]
test_data_bad = data_bad[data_bad['serial_number'].isin(test_serial_bad)]
del data_bad

train_data_good = data_good[data_good['serial_number'].isin(train_serial_good)]
test_data_good = data_good[data_good['serial_number'].isin(test_serial_good)]
del data_good

# get data ready for train of bad
# training_bad = train_data_bad.groupby('serial_number').tail(n=24)
training_bad = train_data_bad
del train_data_bad

# get data ready for train of good
# training_good = train_data_good.groupby('serial_number').apply(lambda df: df.sample(n=4))
training_good = train_data_good
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
random.seed(5)
random.shuffle(index)
training_x_ = training_x[index]
training_y_ = training_y[index]
print('训练数据准备完毕')

# model training
# clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
# clf.fit(training_x_, training_y_)
# gbm = GradientBoostingClassifier(learning_rate=0.01, n_estimators=600, max_depth=7, min_samples_leaf=60,
#                                  min_samples_split=1200, max_features=9, subsample=0.7, random_state=10)
# gbm.fit(training_x_, training_y_)
model = RandomForestClassifier(n_estimators=200, oob_score=True, min_samples_leaf=5, max_depth=10)
model.fit(training_x_, training_y_)
print('模型训练完成')

# save model
with open(files_path + 'rf.pickle', 'wb') as fw:
    pickle.dump(model, fw)

# get data for test
test_bad_x = np.array(test_data_bad.iloc[:, 3:])
test_bad_index = test_data_bad['serial_number']
del test_data_bad

test_good_x = np.array(test_data_good.iloc[:, 3:])
test_good_index = test_data_good['serial_number']
del test_data_good
print('测试数据准备完成')

# y_bad_pre = clf.predict(test_bad_x)
# y_good_pre = clf.predict(test_good_x)
# y_bad_pre = gbm.predict(test_bad_x)
# y_good_pre = gbm.predict(test_good_x)
y_bad_pre = model.predict(test_bad_x)
y_good_pre = model.predict(test_good_x)
print('预测完成\n')

# start to get metrics according to pairs (y_bad_pre, test_bad_index) and (y_good_pre, test_good_index)
num_bad_true = len(test_serial_bad)
num_good_true = len(test_serial_good)

a, serial_place_bad = np.unique(test_bad_index, return_index=True)
a, serial_place_good = np.unique(test_good_index, return_index=True)

# num of predicting bad correctly
num_bad_pre = 0
time_pre = []

serial_place_bad = list(serial_place_bad)
serial_place_bad.append(len(test_bad_index))

for i in range(len(serial_place_bad) - 1):
    sub_pre = y_bad_pre[serial_place_bad[i]:serial_place_bad[i + 1]]
    if sum(sub_pre) > 0:
        num_bad_pre += 1
        time_pre.append(len(sub_pre) - np.where(sub_pre == 1)[0][0] - 1)

# percent of check out for bad disks
FDR = num_bad_pre / num_bad_true

# num of predicting good to bad
num_error_pre = 0
serial_place_good = list(serial_place_good)
serial_place_good.append(len(test_good_index))

for i in range(len(serial_place_good) - 1):
    sub_pre = y_good_pre[serial_place_good[i]:serial_place_good[i + 1]]
    if sum(sub_pre) > 0:
        num_error_pre += 1

# percent of checking good to bad
FAR = num_error_pre / num_good_true

print('坏盘总数为：%d' % num_bad_true)
print('好盘总数为：%d\n' % num_good_true)
print('正确预测出的坏盘数为：%d' % num_bad_pre)
print('好盘为误认为坏盘的数量：%d\n' % num_error_pre)
print('检出率为：%f' % FDR)
print('误检率为：%f\n' % FAR)
print('平均提前时间为：%f\n' % (sum(time_pre) / num_bad_pre))

# Precision,Recall,F1
p = num_bad_pre / (num_bad_pre + num_error_pre)
r = num_bad_pre / num_bad_true
f = 2 * p * r / (p + r)
print('坏盘的精度：%f' % p)
print('坏盘的召回率：%f' % r)
print('坏盘的F1值：%f\n' % f)

p = (num_good_true - num_error_pre) / (num_good_true - num_error_pre + num_bad_true - num_bad_pre)
r = (num_good_true - num_error_pre) / num_good_true
f = 2 * p * r / (p + r)
print('好盘的精度：%f' % p)
print('好盘的召回率：%f' % r)
print('好盘的F1值：%f' % f)

# distribution figure of predicting time in advance
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.figure()
x_list = sorted(list(set(time_pre)))
plt.xticks(x_list)
y_list = []
for x in x_list:
    y_list.append(time_pre.count(x))
plt.bar(x_list, y_list)
for x, y in zip(x_list, y_list):
    plt.text(x, y + 0.1, str(y), ha='center', va='bottom', fontsize=10.5)
plt.title('提前预测时间分布图')
plt.xlabel('提前的天数')
plt.ylabel('被正确预测的坏盘数')
plt.savefig(files_path + 'time_pre.png')
plt.close()

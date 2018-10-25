import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
import pickle


def pettitt_change_point_detection(inputdata):
    inputdata = np.array(inputdata)
    n = inputdata.shape[0]
    k = range(n)
    inputdataT = pd.Series(inputdata)
    r = inputdataT.rank()
    Uk = [2 * np.sum(r[0:x]) - x * (n + 1) for x in k]
    Uka = list(np.abs(Uk))
    U = np.max(Uka)
    K = Uka.index(U)
    pvalue = 2 * np.exp((-6 * (U ** 2)) / (n ** 3 + n ** 2))
    if pvalue <= 0.05:
        change_point_desc = '显著'
    else:
        change_point_desc = '不显著'
    # Pettitt_result = {'突变点位置':K,'突变程度':change_point_desc}
    return K  # ,Pettitt_result


def get_change_point(file):
    df = pd.read_csv(open(file), index_col=0)
    df_bad, df_good = split_bad_good_disks(df)
    bad_disks = list(set(df_bad['serial_number']))
    cols = list(df_bad)
    points = {}
    for col in cols[3:]:
        kk = []
        for disk in bad_disks:
            temp = df_bad[df_bad['serial_number'] == disk]
            k = pettitt_change_point_detection(temp.loc[:, col])
            if k:
                kk.append(len(temp) - k)
        if kk:
            points[col] = int(np.median(kk))
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            mpl.rcParams['axes.unicode_minus'] = False
            plt.figure(figsize=(18, 9))
            # plt.hist(kk)
            plt.title('属性' + col + '  检测到变点的总磁盘数' + str(len(kk)))
            plt.xlabel('变点位置')
            plt.ylabel('磁盘数')
            x_list = sorted(list(set(kk)))
            plt.xticks(x_list)
            y_list = []
            for x in x_list:
                y_list.append(kk.count(x))
            plt.bar(x_list, y_list)
            for x, y in zip(x_list, y_list):
                plt.text(x, y + 0.1, str(y), ha='center', va='bottom', fontsize=10.5)
            plt.savefig('D:\\PycharmProjects\\DiskPrediction\\backblaze\\figure\\snht\\' + col + '.png')
            plt.close()
        else:
            print('属性' + col + '没有检测到变点！')
    return points


if __name__ == '__main__':
    file = 'D:\\PycharmProjects\\DiskPrediction\\backblaze\\data\\ST4000DM000_sort_use_sample.csv'
    change_points = get_change_point(file)
    pickle.dump(change_points, open('D:\\PycharmProjects\\DiskPrediction\\backblaze\\variable\\change_points_snht.pkl', 'wb'))
    change_points = pickle.load(open('D:\\PycharmProjects\\DiskPrediction\\backblaze\\variable\\change_points.pkl', 'rb'))
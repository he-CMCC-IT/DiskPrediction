import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
import pickle
import os

files_path = 'D:\\a学习和工作\\data_Q3_2017\\'


def normalized(df):
    '''
    :param df: 需要标准化的DataFrame磁盘数据
    :return: 标准化到-1到1之间的磁盘数据
    '''

    df_info = df.loc[:, ['date', 'serial_number', 'failure']]
    df = df.iloc[:, 3:]
    df = 2 * (df - df.min()) / (df.max() - df.min()) - 1
    df = pd.concat([df_info, df], axis=1)
    return df


def normalized_2(df):
    '''
    :param df: 需要标准化的DataFrame式磁盘数据
    :return: 标准化到0到1之间的磁盘数据
    '''
    df_info = df.loc[:, ['date', 'serial_number', 'failure']]
    df = df.iloc[:, 3:]
    df = (df - df.min()) / (df.max() - df.min())
    df = pd.concat([df_info, df], axis=1)
    return df


def z_score(df, axis=0):
    '''
    正太分布式标准化，数据需满足正态性假设
    :param df: 需要标准化的DataFrame式磁盘数据
    :param axis: 需要标准化的坐标
    :return: 标准化完的磁盘数据
    '''
    df_info = df.loc[:, ['date', 'serial_number', 'failure']]
    x = df.iloc[:, 3:]
    cols = list(x)
    x = np.array(x).astype(float)
    xr = np.rollaxis(x, axis=axis)
    xr -= np.mean(x, axis=axis)
    xr /= np.std(x, axis=axis)
    # print(x)
    x = pd.DataFrame(x, columns=cols)
    df = pd.concat([df_info, x], axis=1)
    return df


def add_degradation(file, cols, depth=6, nsamples=4):
    '''
    以差分的形式增加时间因素

    参数：
    1. 确定要增加序列信息的数据文件：file是带路径和文件名的变量（已按序列号和时间排序，也没有了空值）。
    2. 确定哪些属性列要求差值：cols是字符串all，就是所有的属性列；
                                   是含有smart子串的字符串，就一个属性列；
                                   是列表，就是列表中的属性列；
                                   是其它，就返回None
    3. 确定做差值的跨度：depth
    4. 确定每个好盘取的样本数：nsamples

    返回值：加上序列信息的DataFrame对象
    '''
    # df = pd.read_csv(open(file), delim_whitespace=True)
    df = pd.read_csv(open(file))

    if isinstance(cols, str):
        if cols.lower() == "all":
            cols = list(df)
            count = 0
            for col in cols:
                if "smart" in col:
                    count += 1
            cols = cols[len(cols) - count:]
        elif "smart" in cols:
            cols = list(cols)
    elif isinstance(cols, list):
        pass
    else:
        return

    serial_number = df['serial_number']
    serial_bad = df[df['failure'] == 1].serial_number
    serial_good = list(set(serial_number).difference(set(serial_bad)))
    serial_bad = list(serial_bad)
    df_failure = df[df['serial_number'].isin(serial_bad)]
    df_success = df[df['serial_number'].isin(serial_good)]

    # 坏盘去掉样本数小于depth的
    df_failure = df_failure.groupby(by=["serial_number"]).apply(
        lambda df_temp: df_temp.sample(n=0) if len(df_temp) <= depth else df_temp)

    # 好盘去掉样本数小于depth+nsamples的（将来还要抽取每个好盘样本数4，不够的话不行）
    df_success = df_success.groupby(by=["serial_number"]).apply(
        lambda df_temp: df_temp.sample(n=0) if len(df_temp) < depth + nsamples else df_temp)

    df_failure.reset_index(drop=True, inplace=True)
    df_success.reset_index(drop=True, inplace=True)
    df = pd.concat([df_failure, df_success], ignore_index=True)

    # 做差分
    df_degrad = pd.concat([df["serial_number"], df[cols]], axis=1)
    df_degrad = df_degrad.groupby(by=['serial_number'], sort=False).apply(lambda df_temp: df_temp.diff(periods=depth))
    df_degrad.dropna(inplace=True)
    df_degrad.reset_index(drop=True, inplace=True)
    df_degrad = np.abs(df_degrad)

    df = df.groupby(by=['serial_number'], sort=False).apply(lambda df_temp: df_temp.iloc[depth:, :])
    df.reset_index(drop=True, inplace=True)

    cols_degrad = []
    for col in cols:
        cols_degrad.append(col + "_degrad")
    df_degrad.columns = cols_degrad

    df = pd.concat([df, df_degrad], axis=1)

    return df


def add_variance(file, cols, depth=12, nsamples=4):
    '''
    以方差的形式增加时间因素

    参数：
    1. 确定要增加序列信息的数据文件：file是带路径和文件名的变量（已按序列号和时间排序，也没有了空值）。
    2. 确定哪些属性列要求差值：cols是字符串all，就是所有的属性列；
                                   是含有smart子串的字符串，就一个属性列；
                                   是列表，就是列表中的属性列；
                                   是其它，就返回None
    3. 确定做方差的样本数：depth
    4. 确定每个好盘取的样本数：nsamples

    返回值：加上序列信息的DataFrame对象
    '''
    df = pd.read_csv(open(file))

    if isinstance(cols, str):
        if cols.lower() == "all":
            cols = list(df)
            count = 0
            for col in cols:
                if "smart" in col:
                    count += 1
            cols = cols[len(cols) - count:]
        elif "smart" in cols:
            cols = list(cols)
    elif isinstance(cols, list):
        pass
    else:
        return

    serial_number = df['serial_number']
    serial_bad = df[df['failure'] == 1].serial_number
    serial_good = list(set(serial_number).difference(set(serial_bad)))
    serial_bad = list(serial_bad)
    df_failure = df[df['serial_number'].isin(serial_bad)]
    df_success = df[df['serial_number'].isin(serial_good)]

    # 坏盘去掉样本数小于depth的
    df_failure = df_failure.groupby(by=["serial_number"]).apply(
        lambda df_temp: df_temp.sample(n=0) if len(df_temp) <= depth - 1 else df_temp)

    # 好盘去掉样本数小于depth+nsamples的（将来还要抽取每个好盘样本数4，不够的话不行）
    df_success = df_success.groupby(by=["serial_number"]).apply(
        lambda df_temp: df_temp.sample(n=0) if len(df_temp) < depth + nsamples - 1 else df_temp)

    df_failure.reset_index(drop=True, inplace=True)
    df_success.reset_index(drop=True, inplace=True)
    df = pd.concat([df_failure, df_success], ignore_index=True)

    # 做方差
    df_degrad = pd.concat([df["serial_number"], df[cols]], axis=1)
    df_degrad = df_degrad.groupby(by=['serial_number'], sort=False).apply(
        lambda df_temp: df_temp.rolling(window=depth).var())
    df_degrad.dropna(inplace=True)
    df_degrad.reset_index(drop=True, inplace=True)

    df = df.groupby(by=['serial_number'], sort=False).apply(lambda df_temp: df_temp.iloc[depth - 1:, :])
    df.reset_index(drop=True, inplace=True)

    cols_var = []
    for col in cols:
        cols_var.append(col + "_var")
    df_degrad.columns = cols_var

    df = pd.concat([df, df_degrad], axis=1)

    return df


def get_train_test_index(serial_number, scale=0.2):
    '''
    按磁盘序列号以8:2的比例随机划分训练集、测试集
    :param serial_number: 全部的磁盘序列号
    :param scale: 划分测试集的比例
    :return: 分别划分到训练集和测试集中的磁盘序列号
    '''
    train_serial = serial_number
    test_serial = []
    test_num = int(len(train_serial) * scale)
    for i in range(test_num):
        random_index = int(np.random.uniform(0, len(train_serial)))
        test_serial.append(train_serial[random_index])
        del train_serial[random_index]
    return train_serial, test_serial


def get_compact_time_series(df, k_list={}, alpha=0.7):
    '''
    得到紧凑的时间序列，属性的变点位置不一样
    :param df: DataFrme, 磁盘数据
    :param k_list: 字典{col: k} = {属性: 变点位置}
    :param alpha: 衰减率
    :return: DataFrame, 紧凑的时间序列
    '''

    def rolling_ewm(sub_df):
        '''
        对每块盘的某些属性做紧凑时间序列处理，每个属性的变点位置不一样
        :param sub_df: 每块盘的数据
        :return: 每块盘的紧凑时间序列表达
        '''
        sub_df.reset_index(drop=True, inplace=True)

        for col, k in k_list.items():
            temp = sub_df[col]
            if len(temp) <= k + 1:
                sub_df.drop(col, axis=1, inplace=True)
                sub_df[col] = temp.ewm(alpha=alpha, adjust=False).mean()
                # sub_df.loc[:, col] = temp.ewm(span=k, adjust=False).mean()
            else:
                sub_temp = temp.iloc[:k + 1].ewm(alpha=alpha, adjust=False).mean()

                a = []
                for j in range(k + 1, len(temp)):
                    a.append(temp.iloc[j - k:j + 1].ewm(alpha=alpha, adjust=False).mean().iloc[-1])
                sub2_temp = pd.DataFrame({col: a})

                sub_df.drop(col, axis=1, inplace=True)
                sub_df[col] = pd.concat([pd.DataFrame(sub_temp), sub2_temp], ignore_index=True)
                # sub_df.loc[:, col] = pd.concat([pd.DataFrame(sub_temp), sub2_temp], ignore_index=True)

        return sub_df

    p = np.where(df['failure'] == 1)[0][-1]
    df_bad = df.iloc[:p + 1, :]
    df_good = df.iloc[p + 1:, :]
    del df

    # for bad disks
    q, p = np.unique(df_bad['serial_number'], return_index=True)
    p = list(p)
    p.append(len(df_bad))

    df_bad_list = []
    count = 0
    for i in range(len(p) - 1):
        df_bad_list.append(rolling_ewm(df_bad.iloc[p[i]:p[i + 1], :]))
        count += 1
        print(count, p[i + 1])
    df_bad = pd.concat(df_bad_list, ignore_index=True)

    # for good disks
    q, p = np.unique(df_good['serial_number'], return_index=True)
    p = list(p)
    p.append(len(df_good))

    df_good_list = []
    count = 0
    for i in range(len(p) - 1):
        df_good_list.append(rolling_ewm(df_good.iloc[p[i]:p[i + 1], :]))
        count += 1
        print(count, p[i + 1])
    df_good = pd.concat(df_good_list, ignore_index=True)

    # concatenation
    df = pd.concat([df_bad, df_good], ignore_index=True)

    return df


def get_compact_time_series_2(file, k, alpha):
    '''
    得到紧凑的时间序列，26个属性的变点位置相同
    :param file: 磁盘数据文件
    :param k: 变点位置
    :param alpha: 衰减率
    :return: DataFrame, 紧凑的时间序列
    '''

    def rolling_ewm(temp):
        '''
        对每块盘的所有属性做紧凑时间序列处理，每个属性的变点位置相同
        :param temp: 每块盘的数据
        :return: 每块盘的紧凑时间序列表达
        '''
        temp.reset_index(drop=True, inplace=True)

        temp_info = temp.iloc[:, :3]
        temp_data = temp.iloc[:, 3:]
        if len(temp_data) <= k + 1:
            temp_data = temp_data.ewm(alpha=alpha, adjust=False).mean()
        else:
            temp_data_1 = temp_data.iloc[:k + 1, :].ewm(alpha=alpha, adjust=False).mean()

            temp_data_2 = pd.concat(
                [temp_data.iloc[j - k:j + 1, :].ewm(alpha=alpha, adjust=False).mean().iloc[-1, :] for j in
                 range(k + 1, len(temp_data))], axis=1)
            temp_data_2 = np.transpose(temp_data_2)
            temp_data_2.reset_index(drop=True, inplace=True)

            temp_data = pd.concat([temp_data_1, temp_data_2], ignore_index=True)

        temp = pd.concat([temp_info, temp_data], axis=1)

        return temp

    df = pd.read_csv(open(file), index_col=0)
    p = np.where(df['failure'] == 1)[0][-1]
    df_bad = df.iloc[:p + 1, :]
    df_good = df.iloc[p + 1:, :]
    del df

    # for bad disks
    q, p = np.unique(df_bad['serial_number'], return_index=True)
    p = list(p)
    p.append(len(df_bad))

    df_bad_list = []
    count = 0
    for i in range(len(p) - 1):
        df_bad_list.append(rolling_ewm(df_bad.iloc[p[i]:p[i + 1], :]))
        count += 1
        print(count, p[i + 1])
    df_bad = pd.concat(df_bad_list, ignore_index=True)

    # for good disks
    q, p = np.unique(df_good['serial_number'], return_index=True)
    p = list(p)
    p.append(len(df_good))

    df_good_list = []
    count = 0
    for i in range(len(p) - 1):
        df_good_list.append(rolling_ewm(df_good.iloc[p[i]:p[i + 1], :]))
        count += 1
        print(count, p[i + 1])
    df_good = pd.concat(df_good_list, ignore_index=True)

    # concatenation
    df = pd.concat([df_bad, df_good], ignore_index=True)

    return df


def plot_nsamples_distribution(file):
    '''
    画故障盘和健康盘的样本数直方图
    :param file: 磁盘数据文件
    :return: 没有返回值
    '''
    df = pd.read_csv(open(file))
    p = np.where(df['failure'] == 1)[0][-1]
    df_bad = df.iloc[:p + 1, :]
    df_good = df.iloc[p + 1:, :]
    del df
    bad_count = df_bad['serial_number'].value_counts()
    good_count = df_good['serial_number'].value_counts()
    plt.bar(range(len(bad_count)), bad_count.values)
    plt.savefig(files_path + 'bad_count.png')
    plt.bar(range(len(good_count)), good_count.values)
    plt.savefig(files_path + 'good_count.png')


def plot_attribute_change_trend(df, att_list, serial_list, figure_path):
    '''
    画每个属性下每块盘岁天数的变化趋势
    :param df: DataFrme格式的磁盘数据
    :param att_list: 属性列表
    :param serial_list: 磁盘序列号列表
    :param figure_path: 保存图的文件路径
    :return: 没有返回值
    '''
    for att in att_list:
        att_path = os.path.join(figure_path, att)
        if not os.path.exists(att_path):
            os.mkdir(att_path)
        for serial in serial_list:
            df_serial = df[df['serial_number'] == serial]
            mpl.rcParams['font.sans-serif'] = ['SimHei']
            plt.figure(figsize=(24, 8))
            plt.plot(range(1, len(df_serial) + 1), list(df_serial[att]))
            plt.plot(range(1, len(df_serial) + 1), list(df_serial[att]), 'r.')
            plt.xticks(range(1, len(df_serial) + 1))
            plt.grid(True)
            plt.xlabel('天')
            plt.ylabel('值')
            plt.title('磁盘' + serial + '的属性' + att + '的变化趋势')
            plt.savefig(os.path.join(att_path, serial + '.png'))
            plt.close()
        print(att)


def sample_good_disk(df=pd.DataFrame([]), n=1000):
    '''
    从健康盘中随机抽部分盘数据
    :param df: DataFrame形式的磁盘数据
    :param n: 抽取健康盘数量
    :return: 经过欠采样后的磁盘数据
    '''
    df_bad, df_good = split_bad_good_disks(df)
    disk_good = list(set(df_good['serial_number']))
    disk_good_sample = []
    for i in range(n):
        random_index = int(np.random.uniform(0, len(disk_good)))
        disk_good_sample.append(disk_good[random_index])
        del disk_good[random_index]
    df_good_sample = df_good[df_good['serial_number'].isin(disk_good_sample)]
    df = pd.concat([df_bad, df_good_sample], ignore_index=True)
    return df


def split_bad_good_disks(df):
    '''
    分隔故障盘和健康盘数据
    :param df: 所有磁盘数据
    :return: 故障盘和健康盘数据
    '''
    p = np.where(df['failure'] == 1)[0][-1]
    df_bad = df.iloc[:p + 1, :]
    df_good = df.iloc[p + 1:, :]
    return df_bad, df_good


def clustering_good_disks(file):
    '''
    对健康盘聚类，还未实现，python函数包不能直接对三维数据聚类
    :param file: 磁盘数据文件
    :return: 空
    '''
    df = pd.read_csv(open(file))
    df_bad, df_good = split_bad_good_disks(df)
    count = df_good['serial_number'].value_counts()
    count = count[count == 92]
    df_good = df_good[df_good['serial_number'].isin(list(count.index))]

    df_good = np.reshape(np.array(df_good.iloc[:, 3:]), (int(len(df_good) / 92), 92, 26))

    estimator = KMeans(n_clusters=10)
    estimator.fit(df_good)
    label_pred = estimator.labels_
    centroids = estimator.cluster_centers_
    # 三维数据不能聚类
    return


def snht_change_point_detection(inputdata):
    '''
    一种变点检测方法
    :param inputdata: 某快磁盘在某个属性上的数据
    :return: 变点位置
    '''
    inputdata = np.array(inputdata)
    inputdata_mean = np.mean(inputdata)
    n = inputdata.shape[0]
    k = range(1, n)
    sigma = np.sqrt(np.sum((inputdata - np.mean(inputdata)) ** 2) / (n - 1))
    Tk = [x * (np.sum((inputdata[0:x] - inputdata_mean) / sigma) / x) ** 2 + (n - x) * (
            np.sum((inputdata[x:n] - inputdata_mean) / sigma) / (n - x)) ** 2 for x in k]
    # print(Tk)
    if not Tk or sigma == 0:
        return 0
    else:
        T = np.max(Tk)
        # print(T)
        K = list(Tk).index(T) + 1
        return K


def pettitt_change_point_detection(inputdata):
    '''
    另一种变点检测方法
    :param inputdata: 某快磁盘在某个属性上的数据
    :return: 变点位置
    '''
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


def get_change_point(df):
    '''
    得到所有故障盘在各个属性上的变点，取所有故障盘变点位置的中位数
    :param df: DataFrame变量，所有磁盘数据
    :return: 字典，{属性: 变点位置}
    '''
    df_bad, df_good = split_bad_good_disks(df)
    bad_disks = list(set(df_bad['serial_number']))
    cols = list(df_bad)
    points = {}
    for col in cols[3:]:
        kk = []
        for disk in bad_disks:
            temp = df_bad[df_bad['serial_number'] == disk]
            # print(disk, col)
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
            plt.savefig(files_path + col + '.png')
            plt.close()
        else:
            print('属性' + col + '没有检测到变点！')
    return points


if __name__ == '__main__':
    start = time.time()
    print('开始时间：%0.2f' % start)

    # normalization of file ST4000DM000_sort_use.csv gotten from get_backblaze_data.py
    data = pd.read_csv(open(files_path + 'ST4000DM000_sort_use.csv'))
    normalized(data).to_csv(files_path + 'ST4000DM000_sort_use_normalized.csv', index=False)

    # sample 1000 good disks randomly and make normalization
    data_sample = sample_good_disk(data, 1000)
    data_sample.to_csv(files_path + 'ST4000DM000_sort_use_sample.csv')
    normalized(data_sample).to_csv(files_path + 'ST4000DM000_sort_use_sample_normalized.csv', index=False)

    # calculate change points
    change_points = get_change_point(data_sample)
    pickle.dump(change_points, open(files_path + 'change_points.pkl', 'wb'))
    # change_points = pickle.load(open(files_path + 'change_points.pkl', 'rb'))

    # get compact time series using different k (i.e. change point) and keeping alpha being 0.7,
    # later continue to make normalization
    data_sample_time_series = get_compact_time_series(data_sample, change_points, 0.7)
    data_sample_time_series.to_csv(files_path + 'ST4000DM000_sort_use_sample_time_series_kDifferent_alpha07.csv',
                                   index=False)
    normalized(data_sample_time_series).to_csv(
        files_path + 'ST4000DM000_sort_use_sample_time_series_kDifferent2_alpha07_normalized.csv', index=False)

    end = time.time()
    print('结束时间：%0.2f' % end)
    print('该函数运行时间为：%0.2f' % (end - start))

    # plot figures of attributes' changing trends for bad and good disks
    '''
    data = pd.read_csv(open(files_path + 'ST4000DM000_sort_use_sample.csv'), index_col=0)
    # normalized(data).to_csv(files_path + 'ST4000DM000_sort_use_sample_normalized.csv', index=False)
    data_bad, data_good = split_bad_good_disks(data)

    figure_bad_path = os.path.join(files_path, 'figure_bad')
    if not os.path.exists(figure_bad_path):
        os.mkdir(figure_bad_path)
    plot_attribute_change_trend(data_bad, data_bad.columns[3:], list(set(data_bad['serial_number'])), figure_bad_path)

    figure_good_path = os.path.join(files_path, 'figure_good')
    if not os.path.exists(figure_good_path):
        os.mkdir(figure_good_path)
    plot_attribute_change_trend(data_good, data_good.columns[3:], list(set(data_good['serial_number']))[0:300],
                                figure_good_path)
    '''

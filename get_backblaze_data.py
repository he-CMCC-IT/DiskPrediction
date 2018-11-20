################################################################################
import pandas as pd
import glob
import os
import configparser

# 从配置文件中读路径
conf = configparser.ConfigParser()
conf.read('conf_backblaze.ini')

path_raw_data = conf.get('config', 'path_raw_data')
path_ST4000DM000_data = conf.get('config', 'path_ST4000DM000_data')
path_ST4000DM000_data_one_csv = conf.get('config', 'path_ST4000DM000_data_one_csv')

# 筛选出model=ST4000DM000的数据，对17年第3季度进行处理
files = os.listdir(path_raw_data)
for filename in files:
    if 'csv' in filename:
        df = pd.read_csv(open(os.path.realpath(filename)))
        df[df['model'] == 'ST4000DM000'].to_csv(path_ST4000DM000_data + filename, index=False)
    print(filename)

# 去除nan太多的属性（先查看一下, 91个文件有相同的42列空值），
# 一共95列，除去不是属性的5列，再除去42列空列，还剩45列属性几乎没有nan值
files = glob.glob(path_ST4000DM000_data + '2017*.csv')
for i in files:
    cols_youzhi = []
    df = pd.read_csv(open(i))
    cols = list(df)
    for col in cols:
        if df[col].count() != 0:
            cols_youzhi.append(col)
    df.loc[:, cols_youzhi].to_csv(i, index=False)

# 合成一个文件
df = pd.concat([pd.read_csv(open(i), index_col=0) for i in files], ignore_index=True)
df.to_csv(path_ST4000DM000_data_one_csv + 'ST4000DM000.csv')

# 坏盘在上（failure都为1），好盘在下（failure都为0），盘的总数是34597，坏盘数是283
df = pd.read_csv(path_ST4000DM000_data_one_csv + 'ST4000DM000.csv', index_col=0)
disks_failure = list(df[df['failure'] == 1].serial_number)
disks_normal = list(set(df['serial_number']).difference(set(disks_failure)))
df_failure = df[df['serial_number'].isin(disks_failure)]
df = df[df['serial_number'].isin(disks_normal)]
# df_failure['failure'] = 1

# 再按序列号和时间排序
df_failure.sort_values(by=['serial_number', 'date'], inplace=True)
df.sort_values(by=['serial_number', 'date'], inplace=True)

# 去掉model和capacity_byte两列，保存到新的文件，坏盘样本数12082，总的样本数3152542(去掉两行坏盘的空记录后变为12080,3152540)
# 容量为137438952960的两行属性都为nan，是Z300GZEK和Z300KYSQ坏点处，其它都没有nan
# 其它的容量都为4000787030016
df = pd.concat([df_failure, df], ignore_index=True)
del df['model']
del df['capacity_bytes']
df.to_csv(path_ST4000DM000_data_one_csv + 'ST4000DM000_sort.csv', index=False)

# 标准化
df = pd.read_csv(open(path_ST4000DM000_data_one_csv + 'ST4000DM000_sort.csv'))

disks_nan = ['Z300GZEK', 'Z300KYSQ']
for disk in disks_nan:
    ind = df[(df['serial_number'] == disk) & (df['failure'] == 1)].index
    print(ind)
    ind = list(ind)[0]
    df.drop(index=ind, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.loc[ind - 1, 'failure'] = 1

# 用kdd论文中的属性
cols_use = ['date', 'serial_number', 'failure', 'smart_1_normalized', 'smart_1_raw', 'smart_5_normalized',
            'smart_5_raw', 'smart_7_normalized', 'smart_7_raw', 'smart_184_normalized', 'smart_184_raw',
            'smart_187_normalized', 'smart_187_raw', 'smart_188_raw', 'smart_189_normalized', 'smart_189_raw',
            'smart_190_normalized', 'smart_190_raw',
            'smart_193_normalized', 'smart_193_raw', 'smart_194_normalized', 'smart_194_raw', 'smart_197_normalized',
            'smart_197_raw', 'smart_198_normalized', 'smart_198_raw', 'smart_240_raw', 'smart_241_raw', 'smart_242_raw']
df = df.loc[:, cols_use]
df_info = df.loc[:, ['date', 'serial_number', 'failure']]
df = df.iloc[:, 3:]
df = 2 * (df - df.min()) / (df.max() - df.min()) - 1
df = pd.concat([df_info, df], axis=1)
df.to_csv(path_ST4000DM000_data_one_csv + 'ST4000DM000_sort_normalized.csv', index=False)
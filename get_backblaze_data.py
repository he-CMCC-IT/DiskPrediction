import pandas as pd
import glob

# 从2017年第3季度的数据中筛选出磁盘模型为ST4000DM000的数据
files_path = 'D:\\a学习和工作\\data_Q3_2017\\'
files = glob.glob(files_path + '\\2017*.csv')
for i in files:
    df = pd.read_csv(open(i))
    df[df['model'] == 'ST4000DM000'].to_csv(files_path + '_' + filename, index=False)

# 去除nan太多的属性（先查看一下, 92个文件有相同的42列空值），
# 一共95列，除去不是属性的5列，再除去42列空列，还剩48列属性几乎没有nan值
files = glob.glob(files_path + '\\_2017*.csv')
for i in files:
    cols_have_value = []
    df = pd.read_csv(open(i))
    cols = list(df)
    for col in cols:
        if df[col].count() != 0:
            cols_have_value.append(col)
    df.loc[:, cols_have_value].to_csv(i, index=False)

# 合成一个文件
df = pd.concat([pd.read_csv(open(i), index_col=0) for i in files], ignore_index=True)
df.to_csv(files_path + 'ST4000DM000.csv')

# 盘的总数是34597，坏盘数是283，好盘数34314
disks_failure = list(df[df['failure'] == 1].serial_number)
disks_normal = list(set(df['serial_number']).difference(set(disks_failure)))
df_failure = df[df['serial_number'].isin(disks_failure)]
df = df[df['serial_number'].isin(disks_normal)]

# 按序列号和时间排序
df_failure.sort_values(by=['serial_number', 'date'], inplace=True)
df.sort_values(by=['serial_number', 'date'], inplace=True)

# 拼接坏盘和好盘，坏盘在上，好盘在下，
# 去掉model和capacity_byte两列，保存到新的文件，总的样本数3152542，坏盘12082，好盘3140460
# 容量为137438952960的两行属性都为nan，是Z300GZEK和Z300KYSQ坏点处，其它都没有nan，其它的容量都为4000787030016
df = pd.concat([df_failure, df], ignore_index=True)
del df['model']
del df['capacity_bytes']
df.to_csv(files_path + 'ST4000DM000_sort.csv', index=False)

# 去除两行空值，总样本数变为3152540，坏盘12080，好盘3140460
disks_nan = ['Z300GZEK', 'Z300KYSQ']
for disk in disks_nan:
    ind = df[(df['serial_number'] == disk) & (df['failure'] == 1)].index
    print(ind)
    ind = list(ind)[0]
    df.drop(index=ind, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.loc[ind - 1, 'failure'] = 1

# 截取26个属性
cols_use = ['date', 'serial_number', 'failure', 'smart_1_normalized', 'smart_1_raw', 'smart_5_normalized',
            'smart_5_raw', 'smart_7_normalized', 'smart_7_raw', 'smart_184_normalized', 'smart_184_raw',
            'smart_187_normalized', 'smart_187_raw', 'smart_188_raw', 'smart_189_normalized', 'smart_189_raw',
            'smart_190_normalized', 'smart_190_raw',
            'smart_193_normalized', 'smart_193_raw', 'smart_194_normalized', 'smart_194_raw', 'smart_197_normalized',
            'smart_197_raw', 'smart_198_normalized', 'smart_198_raw', 'smart_240_raw', 'smart_241_raw', 'smart_242_raw']
df = df.loc[:, cols_use]
df.to_csv(files_path + 'ST4000DM000_sort_use.csv', index=False)

# change_point = {'smart_1_raw': 4, 'smart_5_raw': 12, 'smart_7_raw': 25, 'smart_187_raw': 15, 'smart_197_raw': 10,
#                 'smart_240_raw': 25}

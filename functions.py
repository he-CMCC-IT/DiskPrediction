import pandas as pd


def add_degradation(file, cols, depth=6, nsamples=4):
    '''
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

    df = df.groupby(by=['serial_number'], sort=False).apply(lambda df_temp: df_temp.iloc[depth:, :])
    df.reset_index(drop=True, inplace=True)

    cols_degrad = []
    for col in cols:
        cols_degrad.append(col + "_degrad")
    df_degrad.columns = cols_degrad

    df = pd.concat([df, df_degrad], axis=1)

    return df


df = add_degradation("D:/PycharmProjects/DiskPrediction/backblaze/data/test.csv",
                     ["smart_1_normalized", "smart_1_raw", "smart_3_normalized", "smart_3_raw"])

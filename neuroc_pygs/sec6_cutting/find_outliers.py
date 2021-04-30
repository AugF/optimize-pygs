import numpy as np
import pandas as pd

dir_out = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec6_cutting/exp_thesis_outlier_per'

def box_plot_outliers(data_ser, box_scale=3):
    """
    利用箱线图去除异常值
    :param data_ser: 接收 pandas.Series 数据格式
    :param box_scale: 箱线图尺度，取3; 默认whis=1.5
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html
    :return:
    """
    iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
    # 下阈值
    val_low = data_ser.quantile(0.25) - iqr*0.5
    # 上阈值
    val_up = data_ser.quantile(0.75) + iqr*0.5
    # 异常值
    # outlier = data_ser[(data_ser < val_low) | (data_ser > val_up)]
    up_outlier = data_ser[(data_ser > val_up)]
    # 正常值
    # normal_value = data_ser[(data_ser > val_low) & (data_ser < val_up)]
    return up_outlier


def cal_memory(memory):
    cnt = 0
    if data == 'reddit_sage':
        for i in memory:
            if i > 3 * 1024 * 1024 * 1024:
                cnt += 1
        return cnt
    else:
        for i in memory:
            if i > 2 * 1024 * 1024 * 1024:
                cnt += 1
        return cnt

file_suffix = 'v0'
for data in ['reddit_sage', 'cluster_gcn']:
    xs = []
    if data == 'reddit_sage':
        re_bs = [8700, 8800, 8900]
    else:
        re_bs = [9000, 9100, 9200]
    for bs in re_bs:
        real_path = f'exp_diff_res/{data}_{bs}_{file_suffix}.csv'
        df = pd.read_csv(real_path, index_col=0)
        nodes, edges = df['nodes'], df['edges']
        memory = df['memory']
        up_outliers = cal_memory(memory)
        xs.append(up_outliers / len(memory))
    print(data, xs)

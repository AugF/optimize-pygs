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


file_suffix = 'v1'
for data in ['reddit_sage', 'cluster_gcn']:
    xs = []
    for bs in [1024, 2048, 4096, 8192, 16384]:
        real_path = dir_out + f'/{data}_{bs}_{file_suffix}.csv'
        df = pd.read_csv(real_path, index_col=0)
        nodes, edges = df['nodes'], df['edges']
        memory = df['memory']
        up_outliers = box_plot_outliers(memory)
        xs.append(100 * len(up_outliers) / len(memory))
    print(data, xs)

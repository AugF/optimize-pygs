import numpy as np
import pandas as pd

dir_path = 'out_motivation_data'

for file_name in ['cluster_gcn', 'reddit_sage']:
    for bs in [8400, 8500, 8600, 8700, 8800, 8900]:
        file_path = dir_path + '/' + file_name + '_' + str(bs) + '_v0.csv'
        df = pd.read_csv(file_path, index_col=0)
        cnt = 0
        memory_limit = 3 if file_name == 'reddit_sage' else 2
        for x in df['memory']:
            # print(x)
            if x > memory_limit * 1024 * 1024 * 1024:
                cnt += 1
        print(file_name, bs, cnt / len(df['memory']))



import os
import pandas as pd
from collections import defaultdict

dir_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec6_cutting/exp_opt_res'

dd = defaultdict(list)
for cutting in ['random_0', 'degree_way3', 'degree_way4', 'pagerank_way3', 'pagerank_way4']:
    file_name = dir_path + '/cluster_gcn_opt_' + cutting + '_v0.csv'
    df = pd.read_csv(file_name, index_col=0)
    df.index = df['0']
    del df['0']
    print(df)
    for bs in [9000, 9100, 9200]:
        dd[bs].append([bs] +  df.loc[bs].values.tolist())

for bs in [9000, 9100, 9200]:
    pd.DataFrame(dd[bs]).to_csv(dir_path + f'/cluster_gcn_opt_{bs}_v0.csv')
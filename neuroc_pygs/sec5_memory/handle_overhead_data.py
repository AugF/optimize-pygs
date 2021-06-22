import pandas as pd
from tabulate import tabulate


for predict_model in ['linear_model', 'random_forest']:
    print(predict_model)
    tab_data = []
    for model in ['gcn', 'gat']:
        for data in ['reddit', 'yelp']:
            if model == 'gat' and data == 'reddit':
                re_bs = [170, 175, 180]
            else:
                re_bs = [175, 180, 185]
            for bs in re_bs:
                real_path = f'out_{predict_model}_res/{data}_{model}_{bs}_{predict_model}.csv'
                df = pd.read_csv(real_path, index_col=0)
                total_time, overhead = float(df['total_time'][0]), float(df['overhead'][0])
                tab_data.append([model, data, bs, total_time, overhead])
    print(tabulate(tab_data, headers=['Model', 'Data', 'Batch Size', 'Total Time', 'Overhead'], tablefmt='github'), '\n')
    pd.DataFrame(tab_data, columns=['Model', 'Data', 'Batch Size', 'Total Time', 'Overhead']).to_csv(f'out_train_data/{predict_model}_overhead.csv')
import re
import pandas as pd

file_path = 'sec5_out_2.log'

names = []
for exp_model in ['gcn', 'gat']:
    for predict_model in ['automl', 'linear_model']:
        for exp_data in ['yelp', 'reddit']:
            if exp_data == 'reddit' and exp_model == 'gat':
                re_bs = [170, 175, 180]
            else:
                re_bs = [175, 180, 185]
            for r in re_bs:
                names.append(f'{exp_model}_{predict_model}_{exp_data}_{r}')


df = {}
cnt = 0
with open(file_path) as f:
    for line in f:
        if line.startswith('total'):
            matchline = re.match(r'total time: (.*), overhead: (.*), resampling: (.*), resampling cnt: (.*), info: (.*)', line)
            res = []
            for i in range(1, 6):
                res.append(float(matchline.group(i)))
            print(names[cnt])
            print(res)
            df[names[cnt]] = res + [res[1] - res[2]] + [res[3] / 40]
            cnt += 1

df = pd.DataFrame(df, index=['total', 'overhead', 'resample', 'resampling_cnt', 'info', 'model', 'sampling%'])
linear_df = pd.DataFrame()
for c in df.columns:
    if 'linear' in c:
        linear_df[c] = df[c]

print(linear_df)
linear_df.T.to_csv('sec5_linear_2.csv')
import re
import pandas as pd
from collections import defaultdict

dir_out = 'out_batch_csv'
columns = ['baseline', 'opt', 'real_ratio', 'base_sample', 'base_move', 'base_cal', 'opt_sample', 'opt_move', 'opt_cal', 'y', 'z', 'exp_ratio']

def read_file(file_name, dir_path='out_batch_log'):
    dd_data = defaultdict(list)
    opt_re = r'model: (.*), data: (.*), baseline: (.*), opt:(.*), ratio: (.*)'
    base_time_re = r'Baseline: sample: (.*), move: (.*), cal: (.*)'
    opt_time_re = r'Opt: sample: (.*), move: (.*), cal: (.*)'
    exp_ratio_re = r'y: (.*), z: (.*), exp_ratio: (.*)'
    ratio_cnt, opt_cnt, base_time_cnt, opt_time_cnt = 0, 0, 0, 0
    with open(dir_path + '/' + file_name) as f:
        for line in f:
            ratio_line = re.match(exp_ratio_re, line)
            opt_line = re.match(opt_re, line)
            base_time_line = re.match(base_time_re, line)
            opt_time_line = re.match(opt_time_re, line)
            if ratio_line:
                dd_data[ratio_cnt].extend([float(ratio_line.group(1)), float(ratio_line.group(2)), float(ratio_line.group(3))])
                ratio_cnt += 1
            if opt_line:
                res = [float(opt_line.group(3)), float(opt_line.group(4)), float(opt_line.group(5))]
                dd_data[opt_cnt].extend(res)
                opt_cnt += 1
            if base_time_line:
                dd_data[base_time_cnt].extend([float(base_time_line.group(1)), float(base_time_line.group(2)), float(base_time_line.group(3))])
                base_time_cnt += 1
            if opt_time_line:
                res = [float(opt_time_line.group(1)), float(opt_time_line.group(2)), float(opt_time_line.group(3))]
                dd_data[opt_time_cnt].extend(res)
                opt_time_cnt += 1
    return dd_data


index = []
for exp_data in ['flickr']:
    for exp_model in ['gcn', 'ggnn', 'gaan', 'gat']:
        for var in [1024]:
            index.append(f'{exp_model}_{exp_data}_{var}')

dd_data = read_file('batches_infer_flickr.log')
dd_data = pd.DataFrame(dd_data.values(), index=index, columns=columns)
print(dd_data)
dd_data['1-y-z'] = [1 - dd_data['y'][i] - dd_data['z'][i] for i in dd_data.index]
dd_data['r1'] = dd_data['real_ratio'] / dd_data['exp_ratio'] # 真实的比上期待的，越接近1表示越好
dd_data['y1'] = [dd_data['opt_sample'][i] / (dd_data['opt_sample'][i] + dd_data['opt_move'][i] + dd_data['opt_cal'][i]) for i in dd_data.index]
dd_data['z1'] = [dd_data['opt_move'][i] / (dd_data['opt_sample'][i] + dd_data['opt_move'][i] + dd_data['opt_cal'][i]) for i in dd_data.index]
dd_data['1-y1-z1'] = [dd_data['opt_cal'][i] / (dd_data['opt_sample'][i] + dd_data['opt_move'][i] + dd_data['opt_cal'][i]) for i in dd_data.index]
dd_data['max(y,z,1-y-z)'] = [max(1 - dd_data['y'][i] - dd_data['z'][i], max(dd_data['y'][i], dd_data['z'][i])) for i in dd_data.index]
res = dd_data.reindex(columns=['baseline', 'opt', 'y', 'z', 'max(y,z,1-y-z)', 'exp_ratio', 'real_ratio', 'r1', 'base_sample', 'base_move', 'base_cal', 'opt_sample', 'opt_move', 'opt_cal', 'y', 'z', '1-y-z', 'y1', 'z1', '1-y1-z1'])
res.to_csv(dir_out + f'/batches_infer_flickr.csv')


import time, os
import subprocess
from tabulate import tabulate
from neuroc_pygs.samplers.cuda_prefetcher import CudaDataLoader
from neuroc_pygs.sec4_time.base_epoch import epoch
from neuroc_pygs.configs import PROJECT_PATH

headers = ['Name', 'Baseline', 'Batch Opt', 'Epoch Opt', 'Opt']

def opt_epoch(args=''):
    pro_train = subprocess.Popen(
        "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/opt_train.py " + args, shell=True)
    pro_eval = subprocess.Popen(
        "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/opt_eval.py " + args, shell=True)
    print(f'Pid train process: {pro_train.pid}, eval_process: {pro_eval.pid}')
    pro_eval.communicate()
    pro_eval.wait()
    pro_train.kill()


def run_model(model='gcn', data='amazon-computers'):
    # export MKL_SERVICE_FORCE_INTEL=1
    if not isinstance(model, list):
        model = [model]
    if not isinstance(data, list):
        data = [data]
    
    tab_data = []
    for exp_model in model:
        for exp_data in data:
            args = f'--device cuda:2 --num_workers 0 --model {exp_model} --dataset {exp_data} --epochs 10'
            cur_name = f'{exp_model}_{exp_data}'
            print(cur_name)
            for _ in range(3):
                t1 = time.time()
                opt_epoch(args + ' --opt_train_flag 1 --opt_eval_flag 0') # 优化1+2
                t2 = time.time()
                opt_epoch(args) # 优化2
                t3 = time.time()
                epoch(args + ' --opt_train_flag 1 --opt_eval_flag 0') # 优化1
                t4 = time.time()
                epoch(args) # baseline
                t5 = time.time()
                res = [cur_name, t5 - t4, t4 - t3, t3 - t2, t2 - t1]
                print(res)
                tab_data.append(res) 
        print(tabulate(tab_data, headers=headers, tablefmt='github'))
    return tab_data


if __name__ == '__main__':
    tab_data = run_model()

    small_datasets = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']
    models = ['gcn']
    tab_data = run_model(model=models, data=small_datasets)
    import pandas as pd
    pd.DataFrame(tab_data, columns=headers).to_csv(os.path.join(PROJECT_PATH, 'sec4_time', 'exp_res', f'epoch_v0.csv'))

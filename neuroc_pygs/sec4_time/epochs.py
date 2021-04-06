import time, os
import subprocess
from tabulate import tabulate
from neuroc_pygs.samplers.cuda_prefetcher import CudaDataLoader
from neuroc_pygs.configs import PROJECT_PATH

headers = ['Name', 'Baseline', 'Batch Opt', 'Epoch Opt', 'Opt', 'Batch Ratio%', 'Epoch Raio%', 'Opt%']

def get_ratio(baseline, opt):
    return 100 * (baseline - opt) / baseline

def opt_epoch(args=''):
    print(args)
    pro_train = subprocess.Popen(
        "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/opt_train.py --device cuda:1 " + args, shell=True)
    pro_eval = subprocess.Popen(
        "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/opt_eval.py --device cuda:2 " + args, shell=True)
    print(f'Pid train process: {pro_train.pid}, eval_process: {pro_eval.pid}')
    pro_eval.communicate()
    pro_eval.wait()
    pro_train.kill()


def epoch(args=''):
    print(args)
    pro = subprocess.Popen("python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/base_epoch.py --device cuda:2 " + args, shell=True)
    pro.communicate()
    pro.wait()


def run_model(model='gcn', data='amazon-computers', re_bs=None, mode='cluster'):
    # export MKL_SERVICE_FORCE_INTEL=1
    if not isinstance(model, list):
        model = [model]
    if not isinstance(data, list):
        data = [data]
    if not isinstance(re_bs, list):
        re_bs = [re_bs]
    if not isinstance(mode, list):
        mode = [mode]
    
    tab_data = []
    for exp_model in model:
        for exp_data in data:
            for md in mode:
                for rs in re_bs:
                    default_args = '--hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'
                    args = default_args + f' --num_workers 0 --model {exp_model} --dataset {exp_data} --epochs 50 --mode {md}'
                    if rs != None:
                        args = args + f' --relative_batch_size {rs}'
                    cur_name = f'{exp_model}_{exp_data}_{md}_{rs}'
                    print(cur_name)
                    for _ in range(1):
                        t1 = time.time()
                        opt_epoch(args + ' --opt_train_flag 1 --opt_eval_flag 0') 
                        t2 = time.time()
                        opt_epoch(args) # 优化2
                        t3 = time.time()
                        epoch(args + ' --opt_train_flag 1 --opt_eval_flag 0') # 优化1
                        t4 = time.time()
                        epoch(args) # baseline
                        t5 = time.time()
                        baseline, opt1, opt2, opt12 = t5 - t4, t4 - t3, t3 - t2, t2 - t1
                        ratio1, ratio2, ratio12 = get_ratio(baseline, opt1), get_ratio(baseline, opt2), get_ratio(baseline, opt12)
                        res = [cur_name, baseline, opt1, opt2, opt12, ratio1, ratio2, ratio12]
                        print(res)
                        tab_data.append(res) 
                print(tabulate(tab_data, headers=headers, tablefmt='github'))
    return tab_data


if __name__ == '__main__':
    dir_path = '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/exp_res'
    for files in ['gaan_amazon-computers']:
    # for files in ['gcn_amazon-computers', 'gaan_amazon-computers']:
        exp_model, exp_data = files.split('_')
        for exp_mode in ['cluster']:
            # for exp_pr in [0.01, 0.03, 0.06, 0.1, 0.25, 0.5]:
            # for bs in [64, 128, 256, 512, 1024, 2048]:
            for N in [10, 20, 50, 80, 100, 200]:
            # if True:
                # args = f'--epochs 30 --num_workers 0 --model {exp_model} --dataset {exp_data} --mode {exp_mode} --relative_batch_size {exp_pr}'
                args = f'--epochs {N} --num_workers 0 --model {exp_model} --dataset {exp_data} --mode {exp_mode}'
                # if exp_model == 'gcn':
                #     args = args + f' --hidden_dims {bs}'
                # elif exp_model == 'gaan':
                #     args = args + f' --gaan_hidden_dims {bs}'

                t1 = time.time()
                opt_epoch(args)
                t2 = time.time()
                epoch(args)
                t3 = time.time()
                baseline, opt = t3 - t2, t2 - t1
                ratio = opt / baseline
                print(f'model: {exp_model}, dataset: {exp_data}, mode: {exp_mode}, baseline: {baseline}, opt:{opt}, ratio: {ratio}')

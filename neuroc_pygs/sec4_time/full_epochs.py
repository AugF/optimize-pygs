import time, os
import subprocess
from neuroc_pygs.configs import PROJECT_PATH


def opt_epoch(args=''):
    pro_train = subprocess.Popen(
        "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/opt_full_train.py --device cuda:1 " + args, shell=True)
    pro_eval = subprocess.Popen(
        "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/opt_full_eval.py --device cuda:2 " + args, shell=True)
    print(f'Pid train process: {pro_train.pid}, eval_process: {pro_eval.pid}')
    pro_eval.communicate()
    pro_eval.wait()
    pro_train.kill()


def epoch(args=''):
    pro = subprocess.Popen("python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/base_full_epoch.py " + args, shell=True)
    pro.communicate()
    pro.wait()


if __name__ == '__main__':
    from tabulate import tabulate
    small_datasets = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'flickr']
    tab_data = []
    for model in ['gcn', 'gat']:
        for data in small_datasets:
            args = f'--dataset {data} --model {model} --hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32 --epochs 100'
            try:
                t1 = time.time()
                opt_epoch(args)
                t2 = time.time()
                epoch(args)
                t3 = time.time()
                baseline, opt = t3 - t2, t2 - t1
                ratio = (baseline - opt) / baseline
                print(f'{model}_{data}, baseline: {baseline}, opt: {opt}, ratio: {ratio}')
                tab_data.append([f'{model}_{data}', baseline, opt, ratio])
            except Exception as e:
                print(e)
        print(tabulate(tab_data, headers=['Name', 'Baseline', 'Opt', 'Ratio'], tablefmt='github'))

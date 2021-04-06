import time, os, math
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
    pro = subprocess.Popen("python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/base_full_epoch.py --device cuda:1 " + args, shell=True)
    pro.communicate()
    pro.wait()


def test_all():
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


def test_graph():
    data = 'random'
    model = 'gcn'
    for bs in [5, 10, 20, 40, 80, 85, 90, 95, 100]:
    # for bs in [200, 300, 400, 450, 500, 525, 550, 575]:
        args = f'--epochs 50 --model {model} --dataset random_100k_{bs}k --hidden_dims 2048'
        t1 = time.time()
        opt_epoch(args)
        t2 = time.time()
        epoch(args)
        t3 = time.time()
        baseline, opt = t3 - t2, t2 - t1
        ratio = opt / baseline
        print(f'model: {model}, dataset: {data}, baseline: {baseline}, opt:{opt}, ratio: {ratio}')


def test_batch_size():
    data = 'amazon-computers'
    model = 'gaan'
    for bs in [8, 16, 32, 64, 128, 256, 512, 768, 896]: # 1024, 1088
        args = f'--epochs 50 --model {model} --dataset {data} --gaan_hidden_dims {bs}'
        t1 = time.time()
        opt_epoch(args)
        t2 = time.time()
        epoch(args)
        t3 = time.time()
        baseline, opt = t3 - t2, t2 - t1
        ratio = opt / baseline
        print(f'model: {model}, dataset: {data}, baseline: {baseline}, opt:{opt}, ratio: {ratio}')


def test_epochs():
    data = 'pubmed'
    model = 'gcn'
    for bs in [10, 20, 40, 100, 250, 500]:
        args = f'--epochs {bs} --model {model} --dataset {data}'
        t1 = time.time()
        opt_epoch(args)
        t2 = time.time()
        epoch(args)
        t3 = time.time()
        baseline, opt = t3 - t2, t2 - t1
        ratio = opt / baseline
        print(f'model: {model}, dataset: {data}, baseline: {baseline}, opt:{opt}, ratio: {ratio}')


if __name__ == '__main__':
    data = 'pubmed'
    model = 'gcn'
    for bs in [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]:
        args = f'--epochs 50 --model {model} --dataset {data} --eval_per {bs}'
        t1 = time.time()
        opt_epoch(args)
        t2 = time.time()
        epoch(args)
        t3 = time.time()
        baseline, opt = t3 - t2, t2 - t1
        ratio = opt / baseline
        print(f'model: {model}, dataset: {data}, baseline: {baseline}, opt:{opt}, ratio: {ratio}')


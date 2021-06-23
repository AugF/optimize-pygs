import time, os, math
import subprocess
from neuroc_pygs.configs import PROJECT_PATH


def opt_epoch(args=''):
    pro_train = subprocess.Popen(
        "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/epoch_opt_full_train.py --device cuda:0 " + args, shell=True)
    pro_eval = subprocess.Popen(
        "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/epoch_opt_full_eval.py --device cuda:1 " + args, shell=True)
    print(f'Pid train process: {pro_train.pid}, eval_process: {pro_eval.pid}')
    pro_eval.communicate()
    pro_eval.wait()
    pro_train.kill()


def epoch(args=''):
    pro = subprocess.Popen("python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/epoch_base_full.py --device cuda:0 " + args, shell=True)
    pro.communicate()
    pro.wait()


def run_full(models=['gcn'], datasets=['amazon-computers']):
    for exp_model in models:
        for exp_data in datasets:
            args = f'--dataset {exp_data} --model {exp_model} --hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32 --epochs 100'
            t1 = time.time()
            opt_epoch(args)
            t2 = time.time()
            epoch(args)
            t3 = time.time()
            baseline, opt = t3 - t2, t2 - t1
            ratio = (baseline - opt) / baseline
            print(f'{exp_model}_{exp_data}, baseline: {baseline}, opt: {opt}, ratio: {ratio}')


def run_full_hidden_dims():
    model, data = 'gcn', 'amazon-computers'
    for hd in [64, 128, 256, 512, 1024, 2048, 2304, 2560, 2816, 2944, 3072, 3200, 3328]: 
        args = f'--epochs 50 --model {model} --dataset {data} --hidden_dims {hd}'
        t1 = time.time()
        opt_epoch(args)
        t2 = time.time()
        epoch(args)
        t3 = time.time()
        baseline, opt = t3 - t2, t2 - t1
        ratio = opt / baseline
        print(f'model: {model}, dataset: {data}, baseline: {baseline}, opt:{opt}, ratio: {ratio}')


def run_full_graph():
    model = 'gcn'
    for edges in [5, 10, 20, 40, 80, 85, 90, 95, 100, 200, 300, 400, 450, 500, 525, 550, 575, 600]:
        args = f'--epochs 50 --model {model} --dataset random_100k_{edges}k --hidden_dims 2048'
        t1 = time.time()
        opt_epoch(args)
        t2 = time.time()
        epoch(args)
        t3 = time.time()
        baseline, opt = t3 - t2, t2 - t1
        ratio = opt / baseline
        print(f'model: {model}, edges: {edges}, baseline: {baseline}, opt:{opt}, ratio: {ratio}')


def run_full_N(models='gcn', datasets='pubmed'):
    for N in [10, 20, 40, 100, 250, 500]:
        args = f'--epochs {N} --model {model} --dataset {data}'
        t1 = time.time()
        opt_epoch(args)
        t2 = time.time()
        epoch(args)
        t3 = time.time()
        baseline, opt = t3 - t2, t2 - t1
        ratio = opt / baseline
        print(f'model: {model}, dataset: {data}, N: {N}, baseline: {baseline}, opt:{opt}, ratio: {ratio}')


def test_epoch_full():
    run_full(models=['gcn', 'ggnn', 'gat', 'gaan'], datasets=['amazon-computers', 'flickr'])
    run_full_hidden_dims()
    run_sampling(models=['gcn', 'gaan'], datasets=['pubmed', 'amazon-computers', 'flickr', 'com-amazon'])
    run_full_graph()
    run_full_N(models='gcn', datasets='pubmed')
    run_full_N(models='gaan', datasets='amazon-computers')


if __name__ == '__main__':
    test_epoch_full()

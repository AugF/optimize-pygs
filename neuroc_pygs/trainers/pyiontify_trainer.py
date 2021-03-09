# https://www.cnblogs.com/houjun/p/10407423.html
import time
import os, sys
import subprocess
from neuroc_pygs.trainer import trainer
from neuroc_pygs.configs import ALL_MODELS, EXP_DATASET

# 有效果，甚至很好，丰富测试
def opt_trainer(args):
    pro_train = subprocess.Popen(
        "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/trainers/pyiontify_train.py " + args, shell=True)
    pro_eval = subprocess.Popen(
        "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/trainers/pyiontify_eval.py " + args, shell=True)
    print(f'Pid train process: {pro_train.pid}, eval_process: {pro_eval.pid}')
    pro_eval.communicate()
    pro_eval.wait()
    pro_train.kill()


def run(models='gcn', datasets='pubmed'):
    if not isinstance(models, list):
        models = [models]
    if not isinstance(datasets, list):
        datasets = [datasets]
    for model in models:
        for data in datasets:
            args = ' '.join(['--model', model, '--num_workers', '0', '--dataset', data, '--device', 'cuda:2', '--epochs', '20'])
            sys.argv = [sys.argv[0]] + args.split(' ')
            print('opt trainer')
            t1 = time.time()
            opt_trainer(args)
            t2 = time.time()
            print('base trainer')
            trainer()
            t3 = time.time()
            base_time, opt_time = t3 - t2, t2 - t1
            ratio = 100 * (base_time - opt_time) / base_time
            print(f'{model}, {data}, base_time: {base_time}, opt_time: {opt_time}, ratio: {ratio}')


if __name__ == '__main__':
    run()

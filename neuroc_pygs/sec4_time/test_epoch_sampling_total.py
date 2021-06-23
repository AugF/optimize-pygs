import time, os
import subprocess
from tabulate import tabulate
from neuroc_pygs.configs import PROJECT_PATH

headers = ['Name', 'Baseline', 'Batch Opt', 'Epoch Opt', 'Opt', 'Batch Ratio%', 'Epoch Raio%', 'Opt%']

def get_ratio(baseline, opt):
    return 100 * (baseline - opt) / baseline

def opt_epoch(args=''):
    print(args)
    pro_train = subprocess.Popen(
        "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/epoch_opt_sampling_train.py --device cuda:0 " + args, shell=True)
    pro_eval = subprocess.Popen(
        "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/epoch_opt_sampling_eval.py --device cuda:1 " + args, shell=True)
    print(f'Pid train process: {pro_train.pid}, eval_process: {pro_eval.pid}')
    pro_eval.communicate()
    pro_eval.wait()
    pro_train.kill()


def epoch(args=''):
    print(args)
    pro = subprocess.Popen("python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/epoch_base_sampling.py --device cuda:0 " + args, shell=True)
    pro.communicate()
    pro.wait()


def run_sampling(models=['gcn'], datasets=['amazon-computers'], re_bs=[None], modes=['cluster']):
    for exp_mode in modes:
        for rs in re_bs:
            for exp_model in models:
                for exp_data in datasets:
                    args = f'--epochs 30 --num_workers 0 --model {exp_model} --dataset {exp_data} --mode {exp_mode}'
                    t1 = time.time()
                    opt_epoch(args)
                    t2 = time.time()
                    epoch(args)
                    t3 = time.time()
                    baseline, opt = t3 - t2, t2 - t1
                    ratio = opt / baseline
                    print(f'model: {exp_model}, dataset: {exp_data}, mode: {exp_mode}, baseline: {baseline}, opt:{opt}, ratio: {ratio}')


def run_sampling_hidden_dims():
    exp_model, exp_data, exp_mode = 'gaan', 'amazon-computers', 'cluster'
    for hd in [64, 128, 256, 512, 1024, 2048]:
        args = f'--epochs 30 --num_workers 0 --model {exp_model} --dataset {exp_data} --mode {exp_mode} --gaan_hidden_dims {hd}'
        t1 = time.time()
        opt_epoch(args)
        t2 = time.time()
        epoch(args)
        t3 = time.time()
        baseline, opt = t3 - t2, t2 - t1
        ratio = opt / baseline
        print(f'model: {exp_model}, dataset: {exp_data}, mode: {exp_mode}, baseline: {baseline}, opt:{opt}, ratio: {ratio}')


def run_sampling_N():
    exp_model, exp_data, exp_mode = 'gaan', 'amazon-computers', 'cluster'
    for N in [10, 20, 50, 80, 100, 200]:
        args = f'--epochs {N} --num_workers 0 --model {exp_model} --dataset {exp_data} --mode {exp_mode}'
        t1 = time.time()
        opt_epoch(args)
        t2 = time.time()
        epoch(args)
        t3 = time.time()
        baseline, opt = t3 - t2, t2 - t1
        ratio = opt / baseline
        print(f'model: {exp_model}, dataset: {exp_data}, mode: {exp_mode}, baseline: {baseline}, opt:{opt}, ratio: {ratio}')


def test_epoch_sampling():
    run_sampling(models=['gcn', 'ggnn', 'gat', 'gaan'], datasets=['amazon-computers'])
    run_sampling(models=['gcn'], datasets=['pubmed', 'amazon-computers', 'flickr', 'reddit'])
    run_sampling_hidden_dims()
    run_sampling_N()
    run_sampling(models=['gat'], datasets=['flickr'], re_bs=[0.01, 0.03, 0.06, 0.1, 0.25, 0.5], modes=['cluster', 'graphsage'])


def run_total(models=['gcn'], datasets=['amazon-computers'], re_bs=[None], modes=['cluster']): # 总优化效果测试
    # export MKL_SERVICE_FORCE_INTEL=1
    tab_data = []
    for md in mode:
        for rs in re_bs:
            for exp_model in model:
                for exp_data in data:
                    args = f' --num_workers 0 --model {exp_model} --dataset {exp_data} --epochs 50 --mode {md}'
                    if rs != None:
                        args = args + f' --relative_batch_size {rs}'
                    cur_name = f'{exp_model}_{exp_data}_{md}_{rs}'
                    real_path = 'out_total_csv/'+ cur_name + '_final.csv'
                    print(real_path)
                    if os.path.exists(real_path):
                        tab_data.append(open(real_path).read().split(','))
                        continue
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
                        with open(real_path, 'w') as f:
                            f.write(','.join([str(r) for r in res]))
                        tab_data.append(res) 
                print(tabulate(tab_data, headers=headers, tablefmt='github'))


def test_epoch_total():
    run_total(models=['gcn', 'ggnn', 'gat', 'gaan'], datasets=['pubmed', 'amazon-computers'])
    run_total(models=['gat', 'gaan'], datasets=['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'flickr'])
    run_total(models=['gcn'], datasets=['pubmed'], re_bs=[0.01, 0.03, 0.06, 0.1, 0.25, 0.5], modes=['cluster', 'graphsage'])
    run_total(models=['gaan'], datasets=['amazon-computers'], re_bs=[0.01, 0.03, 0.06, 0.1, 0.25, 0.5], modes=['cluster', 'graphsage'])


if __name__ == '__main__':
    test_epoch_sampling()
    test_epoch_total()
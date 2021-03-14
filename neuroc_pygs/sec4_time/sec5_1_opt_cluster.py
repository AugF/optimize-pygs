import sys
import time
import traceback

from tabulate import tabulate
from neuroc_pygs.samplers.opt_cluster import ClusterOptimizerLoader
from neuroc_pygs.options import build_train_loader, get_args, build_dataset, build_model_optimizer
from neuroc_pygs.configs import EXP_DATASET, EXP_RELATIVE_BATCH_SIZE
from neuroc_pygs.train_step import train

# 三种测试
# 1. 数据集规模测试 EXP_DATASET
# 2. 多线程方案测试
# 3. 不同的Batch Size测试
args = get_args()
args.mode = 'cluster'
# test data
tab_data = []
for num_workers in [5, 10, 20, 40]:
    args.num_workers = num_workers
    for exp_relative_batch_size in EXP_RELATIVE_BATCH_SIZE:
        args.relative_batch_size = exp_relative_batch_size
        for exp_data in EXP_DATASET:
            args.dataset = exp_data 
            file_name = f'num_workers={args.num_workers}, relative_batch_size={args.relative_batch_size}, dataset: {args.dataset}'
            print(file_name)
            try:
                data = build_dataset(args)
                loader1 = build_train_loader(args, data)
                loader2 = build_train_loader(args, data, Cluster_Loader=ClusterOptimizerLoader)
                model, optimizer = build_model_optimizer(args, data)
                model = model.to(args.device)
                t1 = time.time()
                train(model, data, loader1, optimizer, args)
                t2 = time.time()
                train(model, data, loader2, optimizer, args)
                t3 = time.time()
                base_time, opt_time = t2 - t1, t3 - t2
                ratio = 100 * (base_time - opt_time) / base_time
                print(f'base time: {base_time}, opt_time: {opt_time}, ratio: {ratio}')
                tab_data.append([file_name, base_time, opt_time, ratio])
            except Exception as e:
                print(e.args)
                print("======")
                print(traceback.format_exc())


print(tabulate(tab_data, headers=[" ", "Base Time(s)", "Optmize Time(s)", "Ratio(%)"],
        tablefmt="github"))
import os
import time 
import traceback
import numpy as np

from tabulate import tabulate
from neuroc_pygs.samplers.prefetch_generator import BackgroundGenerator
from neuroc_pygs.samplers.data_prefetcher import DataPrefetcher
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer, build_train_loader
from neuroc_pygs.configs import ALL_MODELS, MODES, EXP_DATASET, PROJECT_PATH, EXP_RELATIVE_BATCH_SIZE
from neuroc_pygs.sec4_time.exp2_base_code import train, train_cuda


args = get_args()
print(args)

tab_data = []
tab_data.append(['Name', 'Baseline', 'Opt1', 'Opt2', 'Opt1+Opt2'])
for exp_data in ['pubmed']:
    args.dataset = exp_data
    data = build_dataset(args)
    for exp_model in ALL_MODELS:
        args.model = exp_model
        model, optimizer = build_model_optimizer(args, data)
        for exp_rs in [None] + EXP_RELATIVE_BATCH_SIZE:
            args.relative_batch_size = exp_rs
            for exp_mode in MODES:
                args.mode = exp_mode
                for num_workers in range(0, 36, 5):
                    args.num_workers = num_workers
                    try:
                        train_loader = build_train_loader(args, data)
                        # start 
                        cur_name = f'{args.dataset}_{args.model}_{args.mode}_{args.relative_batch_size}_{args.num_workers}'
                        print(cur_name)
                        model = model.to(args.device)
                        loader_num, device, mode = len(train_loader), args.device, args.mode
                        t1 = time.time()
                        train(model, optimizer, data, iter(train_loader), loader_num, device, mode)
                        t2 = time.time()
                        train(model, optimizer, data, BackgroundGenerator(train_loader), loader_num, device, mode)
                        t3 = time.time()
                        batch_iter = DataPrefetcher(iter(train_loader), mode, device, None if mode == 'cluster' else data)
                        train_cuda(model, optimizer, data, batch_iter, loader_num, device, mode) 
                        t4 = time.time()
                        batch_iter = DataPrefetcher(BackgroundGenerator(train_loader), mode, device, None if mode == 'cluster' else data)
                        train_cuda(model, optimizer, data, batch_iter, loader_num, device, mode) 
                        t5 = time.time()
                        baseline, opt1, opt2, opt12 = t2 - t1, t3 - t2, t4 - t3, t5 - t4
                        opt1_per, opt2_per, opt12_per = (baseline - opt1) / baseline, (baseline - opt2) / baseline, (baseline - opt12) / baseline
                        res = [cur_name, baseline, (opt1, opt1_per), (opt2, opt2_per), (opt12, opt12_per)]
                        print(res)
                        tab_data.append(res)
                    except Exception as e:
                        print(e.args)
                        print("======")
                        print(traceback.format_exc())


np.save(os.path.join(PROJECT_PATH, 'sec4_time', 'exp_res', 'sampling_train.npy'), np.array(tab_data))
print(tabulate(tab_data[1:], headers=tab_data[0], tablefmt='github'))

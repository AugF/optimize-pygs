import torch
import os
import sys
import time 
import traceback
import numpy as np
import pandas as pd

from tabulate import tabulate
from neuroc_pygs.options import get_args, build_dataset, build_model, build_subgraphloader
from neuroc_pygs.configs import ALL_MODELS, MODES, EXP_DATASET, PROJECT_PATH, EXP_RELATIVE_BATCH_SIZE
from neuroc_pygs.samplers.cuda_prefetcher import CudaDataLoader


@torch.no_grad()
def infer(model, data, subgraphloader, split="val"):
    model.eval()
    y_pred = model.inference_base(data.x, subgraphloader)
    y_true = data.y.cpu()

    mask = getattr(data, split + "_mask")
    loss = model.loss_fn(y_pred[mask], y_true[mask])
    acc = model.evaluator(y_pred[mask], y_true[mask]) 
    return acc, loss.item()


@torch.no_grad()
def infer_cuda(model, data, subgraphloader, split="val"):
    model.eval()
    y_pred = model.inference_cuda(data.x, subgraphloader)
    y_true = data.y.cpu()

    mask = getattr(data, split + "_mask")
    loss = model.loss_fn(y_pred[mask], y_true[mask])
    acc = model.evaluator(y_pred[mask], y_true[mask]) 
    return acc, loss.item()


sys.argv = [sys.argv[0], '--device', 'cuda:2', '--num_workers', '0']
args = get_args()
print(args)

small_datasets = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']

tab_data = []
tab_data.append(['Name', 'Baseline', 'Opt', 'Ratio Avg(%)', 'Avg Ratio(%)', 'Max Ratio(%)', 'Min Ratio(%)'])
for exp_data in small_datasets:
    args.dataset = exp_data
    data = build_dataset(args)
    for exp_model in ALL_MODELS:
        args.model = exp_model
        model = build_model(args, data)
        model = model.to(args.device)
        # use
        cur_name = f'{args.dataset}_{args.model}'
        print(cur_name)
        try:
            subgraph_loader = build_subgraphloader(args, data)
            opt_subgraph_loader = CudaDataLoader(subgraph_loader, args.device)

            base_times, opt_times, ratios = [], [], []
            for _ in range(5):
                t1 = time.time()
                infer(model, data, subgraph_loader)
                t2 = time.time()
                infer_cuda(model, data, opt_subgraph_loader)
                t3 = time.time()
                base_time, opt_time = t2 - t1, t3 - t2
                ratio = 100 * (base_time - opt_time) / base_time
                print(f'base time: {base_time}, opt_time: {opt_time}, ratio: {ratio}')
                base_times.append(base_time)
                opt_times.append(opt_time)
                ratios.append(ratio)
            avg_base_time, avg_opt_time = np.mean(base_times), np.mean(opt_times)
            avg_ratio = 100 * (avg_base_time - avg_opt_time) / avg_base_time
            res = [cur_name, avg_base_time, avg_opt_time, np.mean(ratios), avg_ratio, np.max(ratios), np.min(ratios)]
            print(res)
            tab_data.append(res)
        except Exception as e:
            print(e.args)
            print("======")
            print(traceback.format_exc())



pd.DataFrame(tab_data[1:], columns=tab_data[0]).to_csv(os.path.join(PROJECT_PATH, 'sec4_time', 'exp_res', 'sampling_inference_small_datasets_0.csv'))
# np.save(os.path.join(PROJECT_PATH, 'sec4_time', 'exp_res', f'sampling_infer_final.npy'), np.array(tab_data))
print(tabulate(tab_data[1:], headers=tab_data[0], tablefmt='github'))
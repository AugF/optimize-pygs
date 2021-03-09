import torch
import os
import copy
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
def infer(model, data, subgraphloader, df=None):
    model.eval()
    model.reset_parameters()
    y_pred = model.inference_base(data.x, subgraphloader, df)
    y_true = data.y.cpu()

    accs, losses = [], []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        loss = model.loss_fn(y_pred[mask], y_true[mask])
        acc = model.evaluator(y_pred[mask], y_true[mask]) 
        losses.append(loss.item())
        accs.append(acc)
    return accs, losses


sys.argv = [sys.argv[0], '--device', 'cuda:1', '--num_workers', '0']
args = get_args()
print(args)

small_datasets = ['pubmed', 'amazon-photo', 'amazon-computers', 'coauthor-physics', 'flickr', 'com-amazon']

tab_data = []
headers = ['Name', 'Base Sampling', 'Base Transfer', 'Base Training', 'Opt Sampling', 'Opt Transfer', 'Opt Training', 'Base max', 'Base min', 'Opt max', 'Opt min', 'Ratio(%)']
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
            opt_subgraph_loader = CudaDataLoader(copy.deepcopy(subgraph_loader), args.device, sampler='infer_sage')

            loader_num = len(subgraph_loader) * args.layers
            base_times, opt_times = [], []
            for _ in range(50):
                if _ * loader_num >= 50: break
                infer(model, data, subgraph_loader, base_times)
                infer(model, data, opt_subgraph_loader, opt_times)

            base_times, opt_times = np.array(base_times), np.array(opt_times)
            avg_base_times, avg_opt_times, base_all_times, opt_all_times = np.mean(base_times, axis=0), np.mean(opt_times, axis=0), np.sum(base_times, axis=1), np.sum(opt_times, axis=1)
            base_max_time, base_min_time = np.max(base_all_times), np.min(base_all_times)
            base_time, opt_time, opt_max_time, opt_min_time = np.sum(avg_base_times), np.sum(avg_opt_times), np.max(opt_all_times), np.min(opt_all_times)
            print(avg_base_times, avg_opt_times)
            ratio = 100 * (base_time - opt_time) / base_time
            print(f'base time: {base_time}, opt_time: {opt_time}, ratio: {ratio}')
            res = [cur_name, avg_base_times[0], avg_base_times[1], avg_base_times[2], avg_opt_times[0], avg_opt_times[1], avg_opt_times[2], base_max_time, base_min_time, opt_max_time, opt_min_time, ratio]
            tab_data.append(res)
        except Exception as e:
            print(e.args)
            print("======")
            print(traceback.format_exc())


pd.DataFrame(tab_data, columns=headers).to_csv(os.path.join(PROJECT_PATH, 'sec4_time', 'exp_res', f'sampling_inference_final_v1.csv'))
print(tabulate(tab_data, headers=headers, tablefmt='github'))
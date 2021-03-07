import torch
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from neuroc_pygs.options import get_args, build_dataset, build_subgraphloader, build_model
from neuroc_pygs.configs import PROJECT_PATH


@torch.no_grad()
def test(model, data, subgraph_loader, args, df, cnt):
    model.eval()
    device = args.device
    loader_iter, loader_num = iter(subgraph_loader), len(subgraph_loader)
    for i in range(loader_num):
        batch_size, n_id, adjs = next(loader_iter)
        x, y = data.x[n_id], data.y[n_id[:batch_size]]
        x, y = x.to(device), y.to(device)
        adjs = [adj.to(device) for adj in adjs]

        df['nodes'].append(adjs[0][2][0])
        df['edges'].append(adjs[0][0].shape[1])
        logits = model(x, adjs)
        loss = model.loss_fn(logits, y)
        acc = model.evaluator(logits, y) / batch_size

        df['memory'].append(torch.cuda.memory_stats(device)["allocated_bytes.all.peak"])
        torch.cuda.reset_max_memory_allocated(device)
        cnt += 1
        if cnt >= 40:
            break        
    return df, cnt


@torch.no_grad()
def infer(model, data, subgraph_loader, args, df=None, split='test', opt_loader=False):
    device, log_batch, log_batch_dir = args.device, args.log_batch, args.log_batch_dir
    model.eval()
    y_pred = model.inference(data.x, subgraph_loader, log_batch, opt_loader, df=df)
    y_true = data.y.cpu()

    mask = getattr(data, split + "_mask")
    loss = model.loss_fn(y_pred[mask], y_true[mask])
    acc = model.evaluator(y_pred[mask], y_true[mask]) 
    return acc, loss


args = get_args()
args.model = 'gat'
data = build_dataset(args)
model = build_model(args, data)
model = model.to(args.device)
print(args)

for flag in [True, False]: # 剪枝哪一个，边做边看吧
    args.infer_layer = flag
    for bs in [51200, 102400, 204800]:
        args.infer_batch_size = bs
        subgraphloader = build_subgraphloader(args, data)

        file_name = f'{args.model}_{args.dataset}_{args.mode}_{bs}_{flag}'
        print(file_name)
        real_path = os.path.join(PROJECT_PATH, 'sec5_memory/motivation', file_name) + '.csv'
        torch.cuda.reset_max_memory_allocated(args.device)
        if not os.path.exists(real_path):
            if args.infer_layer:  
                res = defaultdict(list)
                num_loader = len(subgraphloader) * args.layers
                for i in range(40):
                    if num_loader * i >= 40:
                        break
                    infer(model, data, subgraphloader, args, df=res)
            else:
                print('start...')
                res = defaultdict(list)
                cnt = 0
                for _ in range(20):
                    res, cnt = test(model, data, subgraphloader, args, res, cnt)
                    if cnt >= 20:
                        break
            memory = np.array(res['memory'])
            print(np.mean(memory), np.median(memory), np.max(memory) - np.min(memory))
            pd.DataFrame(res).to_csv(real_path)

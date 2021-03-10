import torch
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from neuroc_pygs.options import get_args, build_dataset, build_subgraphloader, build_model
from neuroc_pygs.configs import PROJECT_PATH


@torch.no_grad()
def infer(model, data, subgraph_loader, args, df=None, split='test'):
    device, log_batch, log_batch_dir = args.device, args.log_batch, args.log_batch_dir
    model.eval()
    y_pred = model.inference(data.x, subgraph_loader, log_batch, df=df)
    y_true = data.y.cpu()

    mask = getattr(data, split + "_mask")
    loss = model.loss_fn(y_pred[mask], y_true[mask])
    acc = model.evaluator(y_pred[mask], y_true[mask]) 
    return acc, loss


args = get_args()
args.model = 'gat'
print(args)

for exp_data in ['yelp', 'amazon']:
    args.dataset =exp_data
    data = build_dataset(args)
    model = build_model(args, data)
    model = model.to(args.device)
    for bs in [51200, 102400, 204800]:
        args.infer_batch_size = bs
        subgraphloader = build_subgraphloader(args, data)

        file_name = f'inference_{args.model}_{args.dataset}_{args.mode}_{bs}'
        real_path = os.path.join(PROJECT_PATH, 'sec5_memory/motivation', file_name) + '.csv'
        print(real_path)
        torch.cuda.reset_max_memory_allocated(args.device)
        if not os.path.exists(real_path):
            res = defaultdict(list)
            num_loader = len(subgraphloader) * args.layers
            for i in range(40):
                if num_loader * i >= 40:
                    break
                infer(model, data, subgraphloader, args, df=res)
            memory = np.array(res['memory'])
            print(np.mean(memory), np.median(memory), np.max(memory) - np.min(memory))
            pd.DataFrame(res).to_csv(real_path)

import os
import torch
import traceback
import pandas as pd
from collections import defaultdict
from neuroc_pygs.configs import EXP_DATASET, ALL_MODELS, EXP_RELATIVE_BATCH_SIZE, MODES, PROJECT_PATH
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer, build_train_loader


def train(model, data, train_loader, optimizer, args, df, cnt):
    model = model.to(args.device) # special
    device, mode = args.device, args.mode
    model.train()

    loader_iter, loader_num = iter(train_loader), len(train_loader)
    for i in range(loader_num):
        if mode == "cluster":
            # task1
            optimizer.zero_grad()
            batch = next(loader_iter)
            # task2
            batch = batch.to(device)
            df['nodes'].append(batch.x.shape[0])
            df['edges'].append(batch.edge_index.shape[1])
            # task3
            logits = model(batch.x, batch.edge_index)
            loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            acc = model.evaluator(logits[batch.train_mask], batch.y[batch.train_mask])
            optimizer.step()
        else:
            # task1
            batch_size, n_id, adjs = next(loader_iter)
            x, y = data.x[n_id], data.y[n_id[:batch_size]]
            x, y = x.to(device), y.to(device)
            adjs = [adj.to(device) for adj in adjs]
            df['nodes'].append(adjs[0][2][0])
            df['edges'].append(adjs[0][0].shape[1])
            # task3
            logits = model(x, adjs)
            loss = model.loss_fn(logits, y)
            loss.backward()
            acc = model.evaluator(logits, y) / batch_size
            optimizer.step()
        print(f'batch {i}, train_acc: {acc:.4f}, train_loss: {loss.item():.4f}')
        df['memory'].append(torch.cuda.memory_stats(device)["allocated_bytes.all.peak"])
        torch.cuda.reset_max_memory_allocated(device)
        cnt += 1
        if cnt >= 20:
            break
    return df, cnt


def run(model='gcn', dataset='pubmed', mode='cluster', relative_batch_size=None):
    if not isinstance(model, list):
        model = [model]
    if not isinstance(dataset, list):
        dataset = [dataset]
    if not isinstance(mode, list):
        mode = [mode]
    if not isinstance(relative_batch_size, list):
        relative_batch_size = [relative_batch_size]
    run_all(exp_datasets=dataset, exp_models=model, exp_modes=mode, exp_relative_batch_sizes=relative_batch_size)


def run_all(exp_datasets=EXP_DATASET, exp_models=ALL_MODELS, exp_modes=MODES, exp_relative_batch_sizes=EXP_RELATIVE_BATCH_SIZE):
    df = defaultdict(defaultdict)
    args = get_args()
    print(f"device: {args.device}")
    for exp_data in EXP_DATASET:
        args.dataset = exp_data
        data = build_dataset(args)
        print('build data success')
        for exp_model in ALL_MODELS:
            args.model = exp_model
            data = data.to('cpu')
            model, optimizer = build_model_optimizer(args, data)
            print(model)
            for exp_relative_batch_size in EXP_RELATIVE_BATCH_SIZE:
                args.relative_batch_size = exp_relative_batch_size
                for exp_mode in MODES:
                    args.mode = exp_mode
                    data = data.to('cpu')
                    train_loader = build_train_loader(args, data)
                    file_name = '_'.join([args.dataset, args.model, str(args.relative_batch_size), args.mode])
                    torch.cuda.reset_max_memory_allocated(args.device) # 避免dataloader带来的影响
                    print(file_name)
                    real_path = os.path.join(PROJECT_PATH, 'sec5_memory/batch_memory_info', file_name) + '.csv'
                    if os.path.exists(real_path):
                        res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')
                        continue
                    else:
                        try:
                            res = defaultdict(list)
                            cnt = 0
                            for _ in range(20):
                                res, cnt = train(model, data, train_loader, optimizer, args, res, cnt)
                                if cnt >= 20:
                                    break
                            pd.DataFrame(res).to_csv(real_path)
                        except Exception as e:
                            print(e.args)
                            print("======")
                            print(traceback.format_exc())
                    print(res)
                    df[file_name] = res
    return


if __name__ == '__main__':
    run_all()
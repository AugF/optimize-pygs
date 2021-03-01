import os, sys
import torch
import traceback
import pandas as pd
from collections import defaultdict

from neuroc_pygs.options import build_model_optimizer, build_dataset, get_args, build_train_loader

from neuroc_pygs.sec5_memory.configs import MODEL_PARAS
from neuroc_pygs.configs import EXP_DATASET, ALL_MODELS, EXP_RELATIVE_BATCH_SIZE, MODES, PROJECT_PATH


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
            nodes, edges = batch.x.shape[0], batch.edge_index.shape[1]
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
            nodes, edges = adjs[0][2][0], adjs[0][0].shape[1]
            # task3
            logits = model(x, adjs)
            loss = model.loss_fn(logits, y)
            loss.backward()
            acc = model.evaluator(logits, y) / batch_size
            optimizer.step()
        memory = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]
        torch.cuda.reset_max_memory_allocated(device)
        df['nodes'].append(nodes)
        df['edges'].append(edges)
        df['memory'].append(memory)
        print(f'batch: {cnt}, nodes: {nodes}, edges: {edges}, memory: {memory}')
        cnt += 1
        if cnt >= 20:
            break
    return df, cnt


args = get_args()
exp_model = args.model
print(args.device)

x_train = defaultdict(list)
print('build for data')
for exp_data in EXP_DATASET: # dataset nodes, edges
    args.dataset = exp_data
    data = build_dataset(args)
    print("build dataset success")
    model, optimizer = build_model_optimizer(args, data)
    print("build model success")
    for exp_mode in MODES:
        args.mode = exp_mode
        data = data.to('cpu')
        train_loader = build_train_loader(args, data)
        print("build train loader success")
        print('_'.join([args.dataset, args.model, str(args.relative_batch_size), args.mode]))
        torch.cuda.reset_max_memory_allocated(args.device) # 记住一定要清除历史信息
        paras_dict = model.get_hyper_paras() # get other paras
        real_path = os.path.join(PROJECT_PATH, 'sec5_memory/sec5_2_{exp_model}_memory_info', '_'.join([str(v) for v in paras_dict.values()] + [str(args.relative_batch_size), args.mode])) + '.csv'
        print(real_path)
        if os.path.exists(real_path):
            res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')
            for k, v in res.items(): # update x_train
                x_train[k].extend(v)
        else:
            try:
                # 训练集, 每组加20个
                res = defaultdict(list)
                cnt = 0
                for _ in range(20):
                    res, cnt = train(model, data, train_loader, optimizer, args, res, cnt)
                    if cnt >= 20:
                        break
                for key, value in paras_dict.items(): # add x_train
                    res[key].extend([value] * cnt)
                pd.DataFrame(res).to_csv(real_path)
                for k, v in res.items(): # update x_train
                    x_train[k].extend(v)
            except Exception as e:
                print(e.args)
                print(traceback.format_exc())


print("build for other paras")
args.dataset = 'pubmed'
data = build_dataset(args)
print("build dataset success")
for para, para_values in MODEL_PARAS[exp_model].items(): # model, paras
    for p_v in para_values:
        setattr(args, para, p_v)
        data = data.to('cpu')
        model, optimizer = build_model_optimizer(args, data)
        print("build model success")
        for exp_relative_batch_size in EXP_RELATIVE_BATCH_SIZE: # more
            args.relative_batch_size = exp_relative_batch_size
            for exp_mode in MODES:
                args.mode = exp_mode
                data = data.to('cpu')
                train_loader = build_train_loader(args, data)
                print("build train loader success")
                print('_'.join([args.dataset, args.model, str(args.relative_batch_size), args.mode]))
                torch.cuda.reset_max_memory_allocated(args.device) # 记住一定要清除历史信息
                paras_dict = model.get_hyper_paras() # get other paras
                real_path = os.path.join(PROJECT_PATH, f'sec5_memory/sec5_2_{exp_model}_memory_info', '_'.join([str(v) for v in paras_dict.values()] + [str(args.relative_batch_size), args.mode])) + '.csv'
                print(real_path)
                if os.path.exists(real_path):
                    res = pd.read_csv(real_path, index_col=0).to_dict(orient='list')
                    for k, v in res.items(): # update x_train
                        x_train[k].extend(v)
                else:
                    try:
                        # 训练集, 每组加20个
                        res = defaultdict(list)
                        cnt = 0
                        for _ in range(20):
                            res, cnt = train(model, data, train_loader, optimizer, args, res, cnt)
                            if cnt >= 20:
                                break
                        for key, value in paras_dict.items(): # add x_train
                            res[key].extend([value] * cnt)
                        pd.DataFrame(res).to_csv(real_path)
                        for k, v in res.items(): # update x_train
                            x_train[k].extend(v)
                    except Exception as e:
                        print(e.args)
                        print(traceback.format_exc())


pd.DataFrame(x_train).to_csv(PROJECT_PATH + f'/sec5_memory/sec5_2_memory_log/{exp_model}_datasets.csv')

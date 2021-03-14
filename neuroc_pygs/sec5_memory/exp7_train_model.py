import copy
import time
import torch
import numpy as np
import os.path as osp
from neuroc_pygs.train_step import train
from neuroc_pygs.options import prepare_trainer
from neuroc_pygs.configs import PROJECT_PATH


def trainer(model_path): 
    data, train_loader, subgraph_loader, model, optimizer, args = prepare_trainer()
    print(args)
    model = model.to(args.device)
    # step1 fit
    best_val_acc = 0
    best_model = None
    for epoch in range(args.epochs):
        t1 = time.time()
        train_acc, _ = train(model, data, train_loader, optimizer, args) # data由训练负责
        t2 = time.time()
        val_acc, test_acc = infer(model, data, subgraph_loader, args, split="val")
        t3 = time.time()
        # print(f"Epoch: {epoch:03d}, Accuracy: Train: {train_acc:.4f}, Val: {val_acc:.4f}")
        print(f"Epoch: {epoch:03d}, Accuracy: Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

    torch.save(best_model.state_dict(), model_path)
    return


@torch.no_grad()
def infer(model, data, subgraph_loader, args, split="val", opt_loader=False):
    device, log_batch, log_batch_dir = args.device, args.log_batch, args.log_batch_dir
    model.eval()
    y_pred = model.inference(data.x, subgraph_loader, log_batch, opt_loader)
    y_true = data.y.cpu()

    acces = []
    for split in ['val', 'test']:
        mask = getattr(data, split + "_mask")
        acces.append(model.evaluator(y_pred[mask], y_true[mask]) )
    return acces


if __name__ == "__main__":
    import sys, os
    default_args = '--hidden_dims 1024 --gaan_hidden_dims 256 --head_dims 128 --heads 4 --d_a 32 --d_v 32 --d_m 32'
    for data in ['yelp', 'amazon', 'reddit']:
        for model in ['gcn', 'gat']:
            for mode in ['cluster']:
                file_path = osp.join(PROJECT_PATH, f'sec5_memory/exp_inference_cutting/trainer_{model}_{data}_{mode}_best_model.pth')
                print(file_path)
                if os.path.exists(file_path):
                    continue
                sys.argv = [sys.argv[0], '--model', model, '--dataset', data,
                    '--device', 'cuda:0', '--epochs', '1000'] + default_args.split(' ')
                trainer(model_path=file_path)
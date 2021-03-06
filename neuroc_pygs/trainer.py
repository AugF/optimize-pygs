import copy
import time
import torch
import numpy as np
import os.path as osp
from neuroc_pygs.train_step import train, test, infer
from neuroc_pygs.options import prepare_trainer
from neuroc_pygs.configs import PROJECT_PATH


def trainer(): # 训练函数可定制化
    data, train_loader, subgraph_loader, model, optimizer, args = prepare_trainer()
    model = model.to(args.device)
    # step1 fit
    best_val_acc = 0
    best_model = None
    for epoch in range(args.epochs):
        t1 = time.time()
        train_acc, _ = train(model, data, train_loader, optimizer, args) # data由训练负责
        t2 = time.time()
        if args.infer_layer:
            val_acc, _ = infer(model, data, subgraph_loader, args, split="val")
        else:
            val_acc, _ = test(model, data, subgraph_loader, args, split="val")
        t3 = time.time()
        # print(f"Epoch: {epoch:03d}, Accuracy: Train: {train_acc:.4f}, Val: {val_acc:.4f}")
        print(f"Epoch: {epoch:03d}, Accuracy: Train: {train_acc:.4f}, Val: {val_acc:.4f}, train_time: {(t2-t1):.4f}, eval_time:{(t3-t2):.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
    # step2 predict
    if args.infer_layer:
        test_acc, _ = infer(best_model, data, subgraph_loader, args, split="test")
    else:
        test_acc, _ = test(best_model, data, subgraph_loader, args, split="test")
    print(f"final test acc: {test_acc:.4f}")
    torch.save(best_model.state_dict(), osp.join(args.checkpoint_dir, 'trainer_{args.model}_{args.dataset}_best_model.pth'))
    return


if __name__ == "__main__":
    trainer()
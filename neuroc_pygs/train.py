import copy
import time
import numpy as np
import os.path as osp
from neuroc_pygs.train_step import train, test, infer
from neuroc_pygs.options import prepare_trainer
from neuroc_pygs.utils import EpochLogger
from neuroc_pygs.configs import PROJECT_PATH


def trainer(*info, train_func=train, test_func=test, infer_func=infer): # 训练函数可定制化
    data, train_loader, subgraph_loader, model, optimizer, args = info
    
    logger = EpochLogger('_'.join([args.data, args.model, args.mode, str(args.relative_batch_size)]))
    model, data = model.to(args.device), data.to(args.device)

    # step1 fit
    best_val_acc = 0
    best_model = None
    for epoch in range(args.epochs):
        t1 = time.time()
        train_acc, _ = train(model, data, train_loader, optimizer, args)
        t2 = time.time()
        if args.infer_layer:
            val_acc, _ = infer(model, data, subgraph_loader, args, split="val")
        else:
            val_acc, _ = test(model, data, subgraph_loader, args, split="val")
        t3 = time.time()
        if args.log_epoch:
            logger.add_epoch(train_time=t2-t1, eval_time=t3-t2)
        print(f"Epoch: {epoch:03d}, Accuracy: Train: {train_acc:.4f}, Val: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

    if args.log_epoch:
        logger.print_epoch()
        if args.log_epoch_dir:
            logger.save(osp.join(PROJECT_PATH, 'log', args.log_epoch_dir))

    # step2 predict
    if args.infer_layer:
        test_acc, _ = infer(best_model, data, subgraph_loader, args, split="val")
    else:
        test_acc, _ = test(best_model, data, subgraph_loader, args, split="val")
    print(f"final test acc: {test_acc:.4f}")
    return


if __name__ == "__main__":
    trainer(*prepare_trainer())
import copy
import time
import numpy as np

from collections import defaultdict
from tabulate import tabulate

from neuroc_pygs.train_step import train, test, infer
from neuroc_pygs.options import get_args, build_dataset, build_loader, build_model
from neuroc_pygs.opt_train_step import train as opt_train

from neuroc_pygs.configs import ALL_MODELS, EXP_DATASET, MODES


def run(mode, model='gcn', data='pubmed'):
    args = get_args()
    args.model, args.mode, args.data = model, mode, data
    print(args.model, args.mode, args.dataset)
    data = build_dataset(args)
    train_loader, subgraph_loader = build_loader(args, data)
    model, optimizer = build_model(args, data) 
    model, data = model.to(args.device), data.to(args.device)
    # begin test
    t1 = time.time()
    train_acc = train(model, data, train_loader, optimizer, args.mode, args.device)
    t2 = time.time()
    train_acc = opt_train(model, data, train_loader, optimizer, args.mode, args.device)
    t3 = time.time()
    base_time, opt_time = t2 - t1, t3 - t2
    ratio = 100 * (base_time - opt_time) / base_time
    print(base_time, opt_time, ratio)
    return base_time, opt_time, ratio


def run_all():
    tab_data = []
    for mode in MODES:
        for model in ALL_MODELS:
            for data in EXP_DATASET:
                try:
                    base_times, opt_times, ratios = [], [], []
                    for _ in range(5):
                        base_time, opt_time, ratio = run(mode, model, data)
                        base_times.append(base_time)
                        opt_times.append(opt_times)
                        ratios.append(ratio)
                    tab_data.append([(mode, model, data), np.mean(base_times), np.mean(opt_times), np.mean(ratios)])
                except Exception as e:
                    print(e)

    print(tabulate(tab_data, headers=["(Dataset, nodes, edges)", "Base Time(s)", "Optmize Time(s)", "Ratio(%)"],
            tablefmt="github"))


if __name__ == "__main__":
    res = run(mode='graphsage')
    print(res)
    # run_all()
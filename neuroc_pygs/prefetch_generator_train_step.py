# https://zhuanlan.zhihu.com/p/80695364
import time
import numpy as np
import os.path as osp
from neuroc_pygs.options import run, build_loader
from neuroc_pygs.utils import to, BatchLogger
from neuroc_pygs.configs import PROJECT_PATH
from neuroc_pygs.train_step import train
from neuroc_pygs.samplers import ClusterLoaderX, NeighborSamplerX


def func(data, train_loader, subgraph_loader, model, optimizer, args):
    model = model.to(args.device)
    opt_train_loader, _ = build_loader(args, data, Cluster_Loader=ClusterLoaderX, Neighbor_Loader=NeighborSamplerX)
    # begin test
    t1 = time.time()
    train_acc = train(model, data, train_loader, optimizer, args)
    t2 = time.time()
    train_acc = train(model, data, opt_train_loader, optimizer, args)
    t3 = time.time()
    base_time, opt_time = t2 - t1, t3 - t2
    ratio = 100 * (base_time - opt_time) / base_time
    return base_time, opt_time, ratio


if __name__ == '__main__':
    from neuroc_pygs.configs import ALL_MODELS, EXP_DATASET, MODES, EXP_RELATIVE_BATCH_SIZE
    from neuroc_pygs.options import run, run_all 
    import sys
    # sys.argv = [sys.argv[0], '--num_workers', '0']
    # run_all(func, runs=3, path='prefetch_generator_all.out')
    run(func, runs=1, path='prefetch_generator_models.out', model=ALL_MODELS)
    # run(func, runs=1, path='prefetch_generator_dataset.out', dataset=EXP_DATASET)
    # run(func, runs=1, path='prefetch_generator_modes.out', mode=MODES)
    # run(func, runs=1, path='prefetch_generator_rs.out'', mode=EXP_RELATIVE_BATCH_SIZE)
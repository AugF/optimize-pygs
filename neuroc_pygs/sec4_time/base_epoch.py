import sys
import time
from neuroc_pygs.options import get_args, build_dataset, build_train_loader, build_subgraphloader, build_model_optimizer
from neuroc_pygs.sec4_time.epoch_utils import train, infer
from neuroc_pygs.samplers.cuda_prefetcher import CudaDataLoader


def epoch(args):
    sys.argv = [sys.argv[0]] + args.split(' ')
    args = get_args()
    print(args)
    data = build_dataset(args)
    train_loader = build_train_loader(args, data)
    subgraph_loader = build_subgraphloader(args, data)
    if args.opt_train_flag:
        train_loader = CudaDataLoader(train_loader, device=args.device)
    if args.opt_eval_flag:
        subgraph_loader = CudaDataLoader(subgraph_loader, args.device)

    model, optimizer = build_model_optimizer(args, data)
    model = model.to(args.device)
    model.reset_parameters()
    for _ in range(args.epochs):
        t1 = time.time()
        train(model, optimizer, data, train_loader, args.device, args.mode, non_blocking=False)
        t2 = time.time()
        accs, losses = infer(model, data, subgraph_loader)
        t3 = time.time()
        print(f'Epoch: {_:03d}, Train: {accs[0]:.8f}, Val: {accs[1]:.8f}, Test: {accs[2]:.8f}, Train Time: {t2-t1}, Val Time: {t3-t2}')
    return


# if __name__ == '__main__':
#     epoch('--num_workers 0 --opt_eval_flag 1 --opt_train_flag 0')
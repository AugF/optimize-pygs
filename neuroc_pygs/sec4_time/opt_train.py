import os
import torch
import time

from neuroc_pygs.sec4_time.epoch_utils import train
from neuroc_pygs.samplers.cuda_prefetcher import CudaDataLoader
from neuroc_pygs.options import get_args, build_dataset, build_train_loader, build_model_optimizer


def run_train():
    args = get_args()
    print(args)
    data = build_dataset(args)
    train_loader = build_train_loader(args, data)
    if args.opt_train_flag:
        train_loader = CudaDataLoader(train_loader, device=args.device)
    model, optimizer = build_model_optimizer(args, data)
    model = model.to(args.device)
    model.reset_parameters()
    for epoch in range(args.epochs):
        t1 = time.time()
        train(model, optimizer, data, train_loader, args.device, args.mode, non_blocking=False)
        t2 = time.time()
        save_dict = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'train_time': t2 - t1
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(save_dict, os.path.join(args.checkpoint_dir, 'model_%d.pth' % epoch))


if __name__ == '__main__':
    run_train()
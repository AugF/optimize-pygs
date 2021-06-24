import os
import torch
from threading import Thread
from queue import Queue

from neuroc_pygs.sec4_time.epoch_utils import train_full, train
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer, build_train_loader
from neuroc_pygs.samplers.cuda_prefetcher import CudaDataLoader


def train_full(args):
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model, data = model.to(args.device), data.to(args.device)
    for epoch in range(args.epochs):
        train_full(model, data, optimizer)
        save_dict = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(save_dict, os.path.join(args.checkpoint_dir, 'model_%d.pth' % epoch))


def train_sampling(args):
    data = build_dataset(args)
    train_loader = build_train_loader(args, data)
    if args.opt_train_flag:
        train_loader = CudaDataLoader(train_loader, device=args.device)
    model, optimizer = build_model_optimizer(args, data)
    model = model.to(args.device)
    model.reset_parameters()
    for epoch in range(args.epochs):
        train(model, optimizer, data, train_loader, args.device, args.mode, non_blocking=False, opt_flag=args.opt_flag)
        save_dict = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(save_dict, os.path.join(args.checkpoint_dir, 'model_%d.pth' % epoch))


if __name__ == '__main__':
    args = get_args()
    print(args)
    if args.mode == 'None':
        train_full(args)
    else:
        train_sampling(args)
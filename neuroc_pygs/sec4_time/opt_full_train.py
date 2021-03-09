import os
import torch
import time

from neuroc_pygs.sec4_time.epoch_utils import train_full
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer


def run_train():
    args = get_args()
    print(args)
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model, data = model.to(args.device), data.to(args.device)
    for epoch in range(args.epochs):
        t1 = time.time()
        train_full(model, data, optimizer)
        save_dict = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'st_time': t1
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(save_dict, os.path.join(args.checkpoint_dir, 'model_full_%d.pth' % epoch))


if __name__ == '__main__':
    run_train()
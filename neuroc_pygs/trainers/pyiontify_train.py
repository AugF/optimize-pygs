import os
import time
import torch

from neuroc_pygs.train_step import train
from neuroc_pygs.options import get_args, build_dataset, build_train_loader, build_model_optimizer


def run_train():
    args = get_args()
    data = build_dataset(args)
    train_loader = build_train_loader(args, data)
    model, optimizer = build_model_optimizer(args, data)

    model = model.to(args.device)
    # print("begin train")
    for epoch in range(args.epochs):
        t1 = time.time()
        train_acc, train_loss = train(model, data, train_loader, optimizer, args)
        t2 = time.time()
        save_dict = {
            'model_state_dict': model.state_dict(),
            'train_acc': train_acc,
            'train_loss': train_loss,
            'epoch': epoch,
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(save_dict, os.path.join(args.checkpoint_dir, 'model_%d.pth' % epoch))
        t3 = time.time()
        # print(f'Epoch: {epoch:03d}, train_acc: {train_acc: .4f}, train_time: {t2-t1}, overhead_time: {t3-t2}')


if __name__ == '__main__':
    run_train()

import os
import time
import torch

from neuroc_pygs.train_step import train
from neuroc_pygs.options import prepare_trainer


data, train_loader, subgraph_loader, model, optimizer, args = prepare_trainer()
model = model.to(args.device)

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
    print(f'Epoch: {epoch:03d}, train_acc: {train_acc: .4f}, train_time: {t2-t1}, overhead_time: {t3-t2}')

# stop: https://stackoverflow.com/questions/26414052/watch-for-a-file-with-asyncio
# run: https://github.com/LianShuaiLong/CV_Applications/blob/master/classification/classification-pytorch-demo/train.py
import os
import torch
import time
import pyinotify
import asyncio
import copy

from neuroc_pygs.train_step import test, infer
from neuroc_pygs.options import prepare_trainer


data, train_loader, subgraph_loader, model, optimizer, args = prepare_trainer()
model = model.to(args.device)

def evaluate(model_path, best_val_acc):   
    t1 = time.time()
    save_dict = torch.load(model_path)
    model.load_state_dict(save_dict['model_state_dict'])
    t2 = time.time()
    if args.infer_layer:
        val_acc, _ = infer(model, data, subgraph_loader, args, split="val")
    else:
        val_acc, _ = test(model, data, subgraph_loader, args, split="val")
    epoch, train_acc = save_dict['epoch'], save_dict['train_acc']
    t3 = time.time()
    print(f"Epoch: {epoch:03d}, Accuracy: Train: {train_acc:.4f}, Val: {val_acc:.4f}, eval_time: {(t2-t1):.4f}, overhead time: {(t3-t2):.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = copy.deepcopy(model)
    return best_val_acc, epoch == 10


class CREATE_EventHandler(pyinotify.ProcessEvent):
    def my_init(self, loop=None):
        self.loop = loop if loop else asyncio.get_event_loop()
        self.cur_epoch = 0
        self.best_val_acc = 0

    def process_IN_CLOSE_WRITE(self, event):
        newest_file = os.path.join(args.checkpoint_dir, 'model_%d.pth' % self.cur_epoch)
        print(newest_file)
        if os.path.exists(newest_file):
            print('begin to evaluate:{}'.format(newest_file))
            self.best_val_acc, stopping_flag = evaluate(newest_file, self.best_val_acc)
            self.cur_epoch += args.eval_step
            if stopping_flag or self.cur_epoch >= args.epochs:
                self.loop.stop()
        else:
            print('waiting for file ...')


loop = asyncio.get_event_loop()
wm = pyinotify.WatchManager()
mask = pyinotify.IN_CLOSE_WRITE | pyinotify.IN_ACCESS 
handler = CREATE_EventHandler()
wm.add_watch(args.checkpoint_dir, mask)
notifier = pyinotify.AsyncioNotifier(wm, loop, default_proc_fun=handler)
loop.run_forever()
# stop: https://stackoverflow.com/questions/26414052/watch-for-a-file-with-asyncio
# run: https://github.com/LianShuaiLong/CV_Applications/blob/master/classification/classification-pytorch-demo/train.py
import os, sys
import torch
import time
import pyinotify
import asyncio
import copy

from neuroc_pygs.train_step import test, infer
from neuroc_pygs.options import get_args, build_dataset, build_subgraphloader, build_model


def evaluate(model_path, best_val_acc, model, data, subgraph_loader, args):
    # t1 = time.time()
    save_dict = torch.load(model_path)
    model.load_state_dict(save_dict['model_state_dict'])
    # t2 = time.time()
    if args.infer_layer:
        val_acc, _ = infer(model, data, subgraph_loader, args, split="val")
    else:
        val_acc, _ = test(model, data, subgraph_loader, args, split="val")
    epoch, train_acc = save_dict['epoch'], save_dict['train_acc']
    print(f"Epoch: {epoch:03d}, Accuracy: Train: {train_acc:.4f}, Val: {val_acc:.4f}")
    # t3 = time.time()
    # print(f"Epoch: {epoch:03d}, Accuracy: Train: {train_acc:.4f}, Val: {val_acc:.4f}, eval_time: {(t3-t2):.4f}, overhead time: {(t2-t1):.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = copy.deepcopy(model)
    return best_model, best_val_acc, epoch >= 2


class CREATE_EventHandler(pyinotify.ProcessEvent):
    def my_init(self, info, loop=None):
        self.loop = loop if loop else asyncio.get_event_loop()
        self.model, self.data, self.subgraph_loader, self.args = info
        self.cur_epoch = 0
        self.best_val_acc = 0

    def process_IN_CLOSE_WRITE(self, event):  # 名字自取
        newest_file = os.path.join(
            self.args.checkpoint_dir, 'model_%d.pth' % self.cur_epoch)
        if os.path.exists(newest_file):
            self.best_model, self.best_val_acc, stopping_flag = evaluate(
                newest_file, self.best_val_acc, self.model, self.data, self.subgraph_loader, self.args)
            self.cur_epoch += self.args.eval_step
            if stopping_flag or self.cur_epoch >= self.args.epochs:
                if self.args.infer_layer:
                    test_acc, _ = infer(self.best_model, self.data, self.subgraph_loader, self.args, split="test")
                else:
                    test_acc, _ = test(self.best_model, self.data, self.subgraph_loader, self.args, split="test")
                print(f"final test acc: {test_acc:.4f}")
                torch.save(self.best_model.state_dict(), os.path.join(self.args.checkpoint_dir, 'opt_trainer_best_model.pth'))
                self.loop.stop()
                sys.exit(0)
        else:
            print('waiting for file ...')


def run_eval():
    args = get_args()
    args.device = torch.device('cpu')

    data = build_dataset(args)
    subgraph_loader = build_subgraphloader(args, data)
    model = build_model(args, data)
    model = model.to(args.device)
    # print("begin eval")
    loop = asyncio.get_event_loop()
    wm = pyinotify.WatchManager()
    mask = pyinotify.IN_CLOSE_WRITE | pyinotify.IN_ACCESS
    handler = CREATE_EventHandler(
        info=[model, data, subgraph_loader, args], loop=loop)
    wm.add_watch(args.checkpoint_dir, mask)
    notifier = pyinotify.AsyncioNotifier(wm, loop, default_proc_fun=handler)
    loop.run_forever()


if __name__ == '__main__':
    run_eval()

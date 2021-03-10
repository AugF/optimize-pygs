import os, sys
import torch
import time
import pyinotify
import asyncio


from neuroc_pygs.sec4_time.epoch_utils import test_full
from neuroc_pygs.options import get_args, build_dataset, build_model


def evaluate(model_path, model, data):
    t0 = time.time()
    save_dict = torch.load(model_path)
    model.load_state_dict(save_dict['model_state_dict'])
    t1 = time.time()
    test_full(model, data)
    ed_time = time.time()
    epoch, st_time, ed_train_time = save_dict['epoch'], save_dict['st_time'], save_dict['ed_train_time']
    use_time = ed_time - st_time
    overhead_time = t1 - ed_train_time
    print(f'Epoch: {epoch:03d}, opt_train: {t0-st_time}, opt_eval: {ed_time - t0}, opt_train_overhead: {t0-ed_train_time}, opt_eval_overhead: {t1 - t0}, train_time: {ed_train_time-st_time}, eval_time: {ed_time-t1}')

    if epoch == 2:
        return st_time
    else:
        return
    

class CREATE_EventHandler(pyinotify.ProcessEvent):
    def my_init(self, info, loop=None):
        self.loop = loop if loop else asyncio.get_event_loop()
        self.model, self.data, self.args = info
        self.cur_times = []
        self.cur_epoch = 0

    def process_IN_CLOSE_WRITE(self, event):  # 名字自取
        newest_file = os.path.join(
            self.args.checkpoint_dir, 'model_full_%d.pth' % self.cur_epoch)
        if os.path.exists(newest_file):
            res = evaluate(
                newest_file, self.model, self.data)
            if self.cur_epoch == 2:
                self.st_time = res
            self.cur_epoch += 1
            if self.cur_epoch >= self.args.epochs:
                print(f'final use time: {time.time() - self.st_time}')
                self.loop.stop()
                sys.exit(0)
        else:
            print('waiting for file ...')


def run_eval():
    args = get_args()
    data = build_dataset(args)
    model = build_model(args, data)
    model, data = model.to(args.device), data.to(args.device)
    loop = asyncio.get_event_loop()
    wm = pyinotify.WatchManager()
    mask = pyinotify.IN_CLOSE_WRITE | pyinotify.IN_ACCESS
    handler = CREATE_EventHandler(
        info=[model, data, args], loop=loop)
    wm.add_watch(args.checkpoint_dir, mask)
    notifier = pyinotify.AsyncioNotifier(wm, loop, default_proc_fun=handler)
    loop.run_forever()


if __name__ == '__main__':
    run_eval()

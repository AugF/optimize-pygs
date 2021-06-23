import os, sys
import torch
import time
import pyinotify
import asyncio


from neuroc_pygs.sec4_time.epoch_utils import test_full
from neuroc_pygs.options import get_args, build_dataset, build_model


def evaluate(model_path, model, data, eval_per=None):
    t0 = time.time()
    save_dict = torch.load(model_path)
    model.load_state_dict(save_dict['model_state_dict'])
    t1 = time.time()
    epoch, st_time, ed_train_time = save_dict['epoch'], save_dict['st_time'], save_dict['ed_train_time']
    if eval_per is not None: 
        t1 = time.time()
        time.sleep((ed_train_time - st_time) * eval_per / (1 - eval_per))
        t2 = time.time()
    else:
        test_full(model, data)
    ed_time = time.time()
    # use_time = ed_time - st_time
    # overhead_time = t1 - ed_train_time
    print(f'Epoch: {epoch:03d}, opt_train: {t0-st_time}, opt_eval: {ed_time - t0}, opt_train_overhead: {t0-ed_train_time}, opt_eval_overhead: {t1 - t0}, train_time: {ed_train_time-st_time}, eval_time: {ed_time-t1}')

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
            evaluate(
                newest_file, self.model, self.data, self.args.eval_per)
            self.cur_epoch += 1
            if self.cur_epoch >= self.args.epochs:
                self.loop.stop()
                sys.exit(0)
        else:
            print('waiting for file ...')


def run_eval():
    args = get_args()
    if len(os.listdir(args.checkpoint_dir)) > 0:
        os.system(f'rm {args.checkpoint_dir}/*')
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

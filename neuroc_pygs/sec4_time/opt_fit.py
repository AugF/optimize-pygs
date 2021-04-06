import os, sys
import torch
import time
import pyinotify
import asyncio
import copy
import subprocess, math
from threading import Thread
from queue import Queue
from neuroc_pygs.sec4_time.epoch_utils import infer
from neuroc_pygs.samplers.cuda_prefetcher import CudaDataLoader
from neuroc_pygs.options import get_args, build_dataset, build_subgraphloader, build_model
from neuroc_pygs.sec4_time.epoch_utils import train_full
from neuroc_pygs.options import get_args, build_dataset, build_model_optimizer
from neuroc_pygs.configs import PROJECT_PATH


def evaluate(model_path, model, data, subgraph_loader):
    save_dict = torch.load(model_path)
    model.load_state_dict(save_dict['model_state_dict'])
    t1 = time.time()
    accs, losses = infer(model, data, subgraph_loader)
    t2 = time.time()
    epoch, train_time = save_dict['epoch'], save_dict['train_time']
    print(f'Epoch: {epoch:03d}, Train: {accs[0]:.8f}, Val: {accs[1]:.8f}, Test: {accs[2]:.8f}, Train Time: {train_time}, Val Time: {t2-t1}')
    return 


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
            evaluate(
                newest_file, self.model, self.data, self.subgraph_loader)
            self.cur_epoch += 1
            if self.cur_epoch >= self.args.epochs:
                self.loop.stop()
                sys.exit(0)
        else:
            print('waiting for file ...')


def run_eval():
    args = get_args()
    # print(args)
    data = build_dataset(args)
    subgraph_loader = build_subgraphloader(args, data)
    if args.opt_eval_flag:
        subgraph_loader = CudaDataLoader(subgraph_loader, args.device, sampler='infer_sage')

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


def run_train():
    args = get_args()
    print(args)
    data = build_dataset(args)
    model, optimizer = build_model_optimizer(args, data)
    model, data = model.to(args.device), data.to(args.device)
    for epoch in range(args.epochs):
        t1 = time.time()
        train_full(model, data, optimizer)
        t2 = time.time()
        save_dict = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'st_time': t1,
            'ed_train_time': t2
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(save_dict, os.path.join(args.checkpoint_dir, 'model_full_%d.pth' % epoch))


if __name__ == '__main__':
    from multiprocessing import Process
    pro_train = Process(target=run_train) 
    pro_eval = Process(target=run_eval)
    pro_train.start()
    pro_eval.start()
    # pro_train = subprocess.Popen(
    #     "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/opt_full_train.py --device cuda:1 " + args, shell=True)
    # pro_eval = subprocess.Popen(
    #     "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/opt_full_eval.py --device cuda:2 " + args, shell=True)
    # print(f'Pid train process: {pro_train.pid}, eval_process: {pro_eval.pid}')
    # pro_eval.communicate()
    # pro_eval.wait()
    # pro_train.kill()


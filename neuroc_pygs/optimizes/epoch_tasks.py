import time
import torch
import os, sys
import copy

import torch.multiprocessing as mp
from threading import Thread
from code.optimize_epoch.utils import BitSet
from code.optimize_epoch.cora_gcn import model, data, device


tmp_dir = "/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/code/optimize_epoch/tmp"

model_eval, data_eval = copy.deepcopy(model).to(device), copy.deepcopy(data).to(device)
model, data = model.to(device), data.to(device)


def eval_stage(model_dict):
    model_eval.load_state_dict(model_dict)
    train_acc, val_acc, tmp_test_acc = model_eval.eval_step(data_eval)
    print(f"Acc: train {train_acc}, val {val_acc}, tmp_test {tmp_test_acc}")

if __name__ == "__main__":
    epochs = 11
    t1 = time.time()

    jobs = []
    best_val_acc = test_acc = 0
    for epoch in range(1, epochs):
        st0 = time.time()
        loss = model.train_step(data)
        # print(f"epoch: {epoch}, loss: {loss:.4f}")
        st1 = time.time()
        p = Thread(target=eval_stage, args=(model.state_dict(),)) # 多进程，继续尝试多线程
        jobs.append(p)
        p.start()
        st2 = time.time()
        print(f"use time: train_step {st1 - st0}s, overhead {st2 - st1}s")

    for i, j in enumerate(jobs):
        j.join()
        print(i, "time:", time.time() - t1)





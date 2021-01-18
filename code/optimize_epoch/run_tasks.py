import time
import torch
import os
import copy

from multiprocessing import Process
from code.optimize_epoch.utils import BitSet
from code.optimize_epoch.cora_gcn import model, data, device

tmp_dir = "tmp"

model_eval, data_eval = copy.deepcopy(model).to("cpu"), copy.deepcopy(data).to("cpu")
model, data = model.to(device), data.to(device)
loss = model.train_step(data)
print(loss)


t1 = time.time()

def stage1():
    for epoch in range(1, 101):
        print(epoch, "task_training")
        # print(next(model.parameters()).device)
        loss = model.train_step(data)
        torch.save(model.state_dict(), tmp_dir + "/" + str(epoch) + '.pkl')
        os.system(f"echo '{loss}' > {tmp_dir}/{epoch}.log") # 结束标志
        
def stage2():
    results = []
    for epoch in range(1, 101):
        while not os.path.exists(f"{tmp_dir}/{epoch}.log"):
            pass
        model_eval.load_state_dict(torch.load(tmp_dir + "/" + str(args[0]) + '.pkl', map_location=lambda storage, loc: storage))
        accs = model_eval.eval_step(data_eval)
        results.append(accs)
    return results

job1 = Process(target=stage1)
job2 = Process(target=stage2)

job1.start()
job2.start()

job1.join()
job2.join()

best_val_acc = test_acc = 0
for res in job2.results():
    train_acc, val_acc, tmp_test_acc = res
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:04d}, Train: {train_acc:.4f}, '
        f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
        f'Final Test: {test_acc:.4f}')
        
t2 = time.time()

model.reset_parameters()

best_val_acc = test_acc = 0
for epoch in range(1, 101):
    loss = model.train_step(data)
    train_acc, val_acc, tmp_test_acc = model.eval_step(data)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
        f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
        f'Final Test: {test_acc:.4f}')

t3 = time.time()
print(f"pipeline use_time: {t2 - t1}s, origin use_time: {t3 - t2}s")



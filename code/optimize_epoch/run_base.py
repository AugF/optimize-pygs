import time
import torch

from code.optimize_epoch.cora_gcn import model, data, device

tmp_dir = "tmp"
model, data = model.to(device), data.to(device)

# 方法1
t1 = time.time()
best_val_acc = test_acc = 0
for epoch in range(1, 11):
    st0 = time.time()
    loss = model.train_step(data)
    st1 = time.time()
    train_acc, val_acc, tmp_test_acc = model.eval_step(data)
    st2 = time.time()
    print(f"use time: train_step {st1 - st0}s, eval_step {st2 - st1}s")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
        f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
        f'Final Test: {test_acc:.4f}')

t2 = time.time()
print(f"base use time: {t2 - t1}s")
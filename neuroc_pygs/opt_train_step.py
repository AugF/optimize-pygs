import time
import numpy as np
from threading import Thread
from queue import Queue


def train(model, data, train_loader, optimizer, mode, device, log_batch=False):
    loader_iter, loader_num = iter(train_loader), len(train_loader)
    if mode == "cluster":
        return cluster_train_pipeline(loader_num, loader_iter, device, optimizer, model)
    else:
        return graphsage_train_pipeline(loader_num, loader_iter, device, optimizer, model, data)

# 类
def cluster_train_pipeline(loader_num, loader_iter, device, optimizer, model):
    final_loss = []
    def task1(q1):
        for _ in range(loader_num):
            batch = next(loader_iter)
            q1.put(batch)
    
    def task2(q1, q2):
        for _ in range(loader_num):
            batch = q1.get()
            batch = batch.to(device)
            q2.put(batch)
    
    def task3(q2, final_loss):
        for i in range(loader_num):
            batch = q2.get()
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = model.loss_fn(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            final_loss.append(loss.item())
        
    q1, q2 = Queue(maxsize=1), Queue(maxsize=1)
    job1 = Thread(target=task1, args=(q1,))
    job2 = Thread(target=task2, args=(q1, q2, ))
    job3 = Thread(target=task3, args=(q2, final_loss))

    job1.start()
    job2.start()
    job3.start()
    
    job1.join()
    job2.join()
    job3.join()
    return np.mean(final_loss)


def graphsage_train_pipeline(loader_num, loader_iter, device, optimizer, model, data):
    final_loss = []
    def task1(q1):
        for _ in range(loader_num):
            batch = next(loader_iter)
            q1.put(batch)
    
    def task2(q1, q2):
        for _ in range(loader_num):
            batch_size, n_id, adjs = q1.get()
            x, y = data.x[n_id], data.y[n_id[:batch_size]]
            x, y = x.to(device), y.to(device)
            adjs = [adj.to(device) for adj in adjs]
            q2.put((x, y, adjs))
    
    def task3(q2, final_loss):
        for i in range(loader_num):
            x, y, adjs = q2.get()
            optimizer.zero_grad()
            logits = model(x, adjs)
            loss = model.loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            final_loss.append(loss.item())
        
    q1, q2 = Queue(maxsize=1), Queue(maxsize=1)
    job1 = Thread(target=task1, args=(q1,))
    job2 = Thread(target=task2, args=(q1, q2, ))
    job3 = Thread(target=task3, args=(q2, final_loss))

    job1.start()
    job2.start()
    job3.start()
    
    job1.join()
    job2.join()
    job3.join()
    return np.mean(final_loss)
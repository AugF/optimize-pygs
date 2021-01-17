import time
from threading import Thread
from queue import Queue

import torch.nn.functional as F

#
def train_thread_3(model, loader, optimizer, device):
    model.train()
    num = len(loader)
    
    st1 = time.time()
    def task1(q1):
        loader_iter = iter(loader)
        for i in range(num):
            data = next(loader_iter)
            time.sleep(2)
            q1.put(data)
    
    def task2(q1, q2):
        for i in range(num):
            data = q1.get()
            if i == 0:
                print(f"task2 begin: {time.time() - st1}s")
            data = data.to(device)
            time.sleep(1)
            q2.put(data)
        
    def task3(q2):
        total_loss = total_examples = total_correct = 0
        for i in range(num):
            data = q2.get()
            if i == 0:
                print(f"task3 begin: {time.time() - st1}s")
            if data.train_mask.sum() == 0: # task3
                continue
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)[data.train_mask]
            y = data.y.squeeze(1)[data.train_mask]
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()

            num_examples = data.train_mask.sum().item()
            total_loss += loss.item() * num_examples
            total_examples += num_examples

            total_correct += out.argmax(dim=-1).eq(y).sum().item()
            total_examples += y.size(0)
            time.sleep(4)
        print(f'loss: {total_loss / total_examples:.4f}, train_acc: {total_correct / total_examples:.4f}')

    q1, q2 = Queue(maxsize=1), Queue(maxsize=1)
    job1 = Thread(target=task1, args=(q1,))
    job2 = Thread(target=task2, args=(q1, q2, ))
    job3 = Thread(target=task3, args=(q2, ))
    
    job1.start()
    job2.start()
    job3.start()
    
    job1.join()
    st2 = time.time()
    job2.join()
    st3 = time.time()
    job3.join()
    st4 = time.time()
    print(f"total_batch: {num}, total sampling_time: {st2 - st1}s, total to_time: {st3 - st1}s, total training_time: {st4 - st1}s")
    return


def train_queue_3(model, loader, optimizer, device):
    model.train()
    num = len(loader)
    
    def task1(q1):
        loader_iter = iter(loader)
        for i in range(num):
            data = next(loader_iter)
            time.sleep(2)
            q1.put(data)
    
    def task2(q1, q2):
        for i in range(num):
            data = q1.get()
            if i == 0:
                print(f"task2 begin: {time.time() - st1}s")
            data = data.to(device)
            time.sleep(1)
            q2.put(data)
        
    def task3(q2):
        total_loss = total_examples = total_correct = 0
        for i in range(num):
            data = q2.get()
            if i == 0:
                print(f"task3 begin: {time.time() - st1}s")
            if data.train_mask.sum() == 0: # task3
                continue
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)[data.train_mask]
            y = data.y.squeeze(1)[data.train_mask]
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()

            num_examples = data.train_mask.sum().item()
            total_loss += loss.item() * num_examples
            total_examples += num_examples

            total_correct += out.argmax(dim=-1).eq(y).sum().item()
            total_examples += y.size(0)
            time.sleep(4)
        print(f'loss: {total_loss / total_examples:.4f}, train_acc: {total_correct / total_examples:.4f}')

    q1, q2 = Queue(maxsize=1), Queue(maxsize=1)
    job1 = Thread(target=task1, args=(q1,))
    job2 = Thread(target=task2, args=(q1, q2, ))
    job3 = Thread(target=task3, args=(q2, ))
    
    job1.start()
    job2.start()
    job3.start()
    
    job1.join()
    st2 = time.time()
    job2.join()
    st3 = time.time()
    job3.join()
    st4 = time.time()
    print(f"total_batch: {num}, total sampling_time: {st2 - st1}s, total to_time: {st3 - st1}s, total training_time: {st4 - st1}s")
    return
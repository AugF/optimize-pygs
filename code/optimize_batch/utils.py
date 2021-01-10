import torch
import time
import argparse
from threading import Thread
from queue import Queue
import torch.nn.functional as F

def get_args(description='OGBN-Products'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_partitions', type=int, default=15000)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    
    return args

def train_base(model, loader, optimizer, device):
    model.train()

    total_loss = total_examples = 0
    total_correct = total_examples = 0
    
    num = len(loader)
    loader_iter = iter(loader)
    
    sampling_time, to_time, training_time = 0, 0, 0
    for i in range(num):
        t1 = time.time()
        data = next(loader_iter)
        time.sleep(2) # add
        t2 = time.time()
        data = data.to(device)
        time.sleep(1) # add
        t3 = time.time()
        if data.train_mask.sum() == 0:
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
        time.sleep(4) # add
        t4 = time.time()
        print(f"batch {i}: sampling_time: {t2 - t1}s, to_time: {t3 - t2}s, training_time: {t4 - t3}s")
        sampling_time += t2 - t1
        to_time += t3 - t2
        training_time += t4 - t3

    print(f"total sampling_time: {sampling_time}s, total to_time: {to_time}s, total training_time: {training_time}s")
    list1 = [sampling_time / num, to_time / num, training_time / num]
    expect_time = max(list1) * (num - 1) + sum(list1)
    print(f'total_batch: {num}, expect pipeline time: {expect_time}s')
    print(f'loss: {total_loss / total_examples:.4f}, train_acc: {total_correct / total_examples:.4f}')
    return


@torch.no_grad()
def test_base(model, data, evaluator, subgraph_loader, device):
    model.eval()

    out = model.inference(data.x, subgraph_loader, device)

    y_true = data.y
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[data.train_mask],
        'y_pred': y_pred[data.train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[data.valid_mask],
        'y_pred': y_pred[data.valid_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.test_mask],
        'y_pred': y_pred[data.test_mask]
    })['acc']

    return train_acc, valid_acc, test_acc

def train_thread_2(model, loader, optimizer, device):
    model.train()
    num = len(loader)
    
    def task1(q1):
        loader_iter = iter(loader)
        for i in range(num):
            data = next(loader_iter)
            q1.put(data.to(device, non_blocking=True))
        
    def task2(q1):
        total_loss = total_examples = total_correct = 0
        for i in range(num):
            data = q1.get()
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
        print(f'loss: {total_loss / total_examples:.4f}, train_acc: {total_correct / total_examples:.4f}')

    q1 = Queue()
    job1 = Thread(target=task1, args=(q1,))
    job2 = Thread(target=task2, args=(q1, ))
    
    job1.start()
    job2.start()
    
    job1.join()
    job2.join()
    return


class MyThread(Thread): # 相比全局变量，有点慢
    def __init__(self, target, args):
        super(MyThread, self).__init__()
        self.target = target
        self.args = args

    def run(self):
        self.result = self.target(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
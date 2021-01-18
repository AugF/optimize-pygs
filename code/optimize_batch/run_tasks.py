import time
from code.optimize_batch.tasks import num, model, optimizer, device, loader, task1, task2, task3
import multiprocessing as mp

def job1(q):
    loader_iter = iter(loader)
    for i in range(num):
        print(f"begin {i}_task1")
        data_cpu = task1(loader_iter)
        q.put(data_cpu)


def job2(q, subq, device):
    for i in range(num):
        data_cpu = q.get()
        print(f"begin {i}_task2")
        data_gpu = task2(data_cpu, device)
        subq.put(data_gpu)


def job3(subq, model, optimizer):
    total_loss, total_acc, total_num = 0.0, 0, 0
    for i in range(num):
        data_gpu = subq.get()
        print(f"begin {i}_task3")
        res = task3(data_gpu, model, optimizer)
        model, optimizer = res[0], res[1]
        total_loss += res[2]
        total_acc += res[3]
        total_num += res[4]
    return total_loss, total_acc, total_num


print("begin")
# mp.set_start_method('spawn')
q, subq = mp.Queue(1), mp.Queue(1)
jobs = []
jobs.append(mp.Process(target=job1, args=(q,)))
jobs.append(mp.Process(target=job2, args=(q, subq, device)))
jobs.append(mp.Process(target=job3, args=(subq, model, optimizer)))

# start
for j in jobs:
    j.start()

# wait
for j in jobs:
    j.join()

total_loss, total_acc, total_num = jobs[-1].recv()
print(f"total_loss={total_loss}, total_acc={total_acc}, total_num={total_num}")
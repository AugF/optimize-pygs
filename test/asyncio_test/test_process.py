from multiprocessing import Process, Queue
import os, time, random

use_time_g1 = []
use_time_g2 = []
use_time_g3 = []

def g1(step):
    st = time.time()
    x = 1
    for i in range(200000):
        x *= i
    use_time_g1.append((time.time() - st) * 1000)

def g2(step):
    st = time.time()
    x = 1
    for i in range(100000):
        x *= i
    use_time_g2.append((time.time() - st) * 1000)


def g3(step):
    st = time.time()
    x = 1
    for i in range(400000):
        x *= i
    use_time_g3.append((time.time() - st) * 1000)

def fun1(out_q):
    print(f"begin fun1: {(time.time() - t0) * 1000}s")
    for i in range(num):
        g1(i)
        data = random.ranint(0, 4)
        out_q.put(data)
    
def fun2(in_q, sub_q):
    print(f"begin fun2: {(time.time() - t0) * 1000}s")
    for i in range(num):
        data = in_q.get()
        g2(i)
        sub_q.put(data)

def fun3(in_q):
    print(f"begin fun3: {(time.time() - t0) * 1000}s")
    for i in range(num):
        data = in_q.get()
        g3(i)
        
num = 3
q = Queue()
sub_q = Queue()
task1 = Process(target=fun1, args=(q,))
task2 = Process(target=fun2, args=(q, sub_q, ))
task3 = Process(target=fun3, args=(sub_q,))
t0 = time.time()
task1.start()
task2.start()
task3.start()

task1.join()
task2.join()
task3.join()
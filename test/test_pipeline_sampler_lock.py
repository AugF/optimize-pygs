"""
sampler, cuda, training
"""
import time
import random
from threading import Thread
from queue import Queue

# from multiprocessing import Process, Queue

def producer(out_q):
    for i in range(num):
        time.sleep(st1)
        data = random.randint(0, 10)
        print(f"producer produce {data}")
        out_q.put(data)
    
def consumer(in_q, sub_q):
    for i in range(num):
        data = in_q.get()
        print(f'consumer get {data}')
        time.sleep(st2)
        sub_q.put(data)

def sub_consumer(in_q):
    for i in range(num):
        data = in_q.get()
        print(f'sub_consumer get {data}')
        time.sleep(st3)    

for i in range(3):
    # set 
    st1, st2, st3 = random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)
    num = 3
    print(st1, st2, st3, num)
    # run
    st = time.time()    
    q = Queue()
    sub_q = Queue()
    t1 = Thread(target=producer, args=(q,))
    t2 = Thread(target=consumer, args=(q, sub_q, ))
    t3 = Thread(target=sub_consumer, args=(sub_q,))
    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()
    print(f"expect time: {max(st1, max(st2, st3)) * (num - 1) + sum([st1, st2, st3])}s, use time: {time.time() - st}s")


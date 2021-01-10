"""
sampler, cuda, training
"""
import time
import random
from threading import Thread
from queue import Queue

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

# 100ms = 0.1s, 10ms级别没有用，100ms级别有用
for i in range(1):
    # set 
    st1, st2, st3 = 0.05, 0.01, 0.02
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
    print(f"max_time: {sum([st1, st2, st3]) * num}s, expect time: {max(st1, max(st2, st3)) * (num - 1) + sum([st1, st2, st3])}s, use time: {time.time() - st}s")


"""
sampler, cuda, training
"""
import time
import random
from threading import Thread
from queue import Queue

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

def producer(out_q):
    for i in range(num):
        time.sleep(st1)
        data = random.randint(0, 10)
        print(f"producer produce {data}")
        out_q.put(data)
    
def consumer(in_q):
    x = 1
    for i in range(num):
        data = in_q.get()
        print(f'consumer get {data}')
        time.sleep(st2)
        x += data
    return x

for i in range(1):
    # set 
    st1, st2 = random.randint(1, 5), random.randint(1, 5)
    num = 3
    print(f'st1={st1}, st2={st2}, num={num}')
    # run
    st = time.time()    
    q = Queue()
    task1 = Thread(target=producer, args=(q,))
    task2 = MyThread(target=consumer, args=(q,))
    task1.start()
    task2.start()

    task1.join()
    task2.join()
    print(task2.get_result())
    print(f"orgin time: {(st1 + st2) * num}s, expect time: {max(st1, st2) * (num - 1) + sum([st1, st2])}s, use time: {time.time() - st}s")


from threading import *
import time
import logging

# full_cpu_sem: 初始化为0，表示有数据，可以进行操作;
full_cpu_sem, empty_cpu_sem, full_cuda_sem, empty_cuda_sem = Semaphore(0), Semaphore(1), Semaphore(0), Semaphore(1)
cpu_lock, cuda_lock = Lock(), Lock()
data_cpu, data_gpu = None, None
use_time_g1 = []
use_time_g2 = []
use_time_g3 = []
timelines_f1 = []
timelines_f2 = []
timelines_f3 = []

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

def fun1():
    global data_cpu, data_gpu
    print(f"fun1 start use time: {(time.time() - t0) * 1000}ms")
    for i in range(num):
        g1(i)
        fun1_time = (time.time() - t0) * 1000
        timelines_f1.append(fun1_time)

        if i == 0:
            print(f"fun1 begin sem use time: {(time.time() - t0) * 1000}ms")
        empty_cpu_sem.acquire()   # 1
        if i == 0:
            print(f"fun1 begin lock use time: {(time.time() - t0) * 1000}ms")
        with cpu_lock: # 写必须阻塞
            data_cpu = i
        if i == 0:
            print(f"fun1 end lock use time: {(time.time() - t0) * 1000}ms")
        full_cpu_sem.release()
        if i == 0:
            print(f"fun1 end sem use time: {(time.time() - t0) * 1000}ms")

def fun2():
    global data_cpu, data_gpu
    print(f"fun2 start use time: {(time.time() - t0) * 1000}ms")
    for i in range(num):
        if i == 0:
            print(f"fun2 begin use time: {(time.time() - t0) * 1000}ms")
        full_cpu_sem.acquire() # 0
        if i == 0:
            print(f"fun2 begin lock use time: {(time.time() - t0) * 1000}ms")
        with cpu_lock: # 读需要阻塞吗？
            x = data_cpu
        if i == 0:
            print(f"fun2 end lock use time: {(time.time() - t0) * 1000}ms")
        empty_cpu_sem.release()
        
        if i == 0:
            print(f"use time: {(time.time() - t0) * 1000}ms")

        g2(i)
        fun2_time = (time.time() - t0) * 1000
        timelines_f2.append(fun2_time)

        empty_cuda_sem.acquire() # 1
        with cuda_lock:
            data_gpu = x + 1
        full_cuda_sem.release() # data_cuda已经产生

def fun3():
    global data_cpu, data_gpu
    print(f"fun3 start use time: {(time.time() - t0) * 1000}ms")
    for i in range(num):
        full_cuda_sem.acquire() # 0
        with cuda_lock:
            x = data_gpu
        empty_cuda_sem.release()  
        g3(i)
        fun3_time = (time.time() - t0) * 1000
        timelines_f3.append(fun3_time)
        # print(f"finsh {i} step fun3 use time: {fun3_time}ms")

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                datefmt="%H:%M:%S")    

num = 2
task1 = Thread(target=fun1)
task2 = Thread(target=fun2)
task3 = Thread(target=fun3)

t0 = time.time()
task1.start()
task2.start()
task3.start()

task1.join()
print(f"task1 use time: {(time.time() - t0) * 1000}ms")

task2.join()
print(f"task2 use time: {(time.time() - t0) * 1000}ms")

task3.join()
print(f"task3 use time: {(time.time() - t0) * 1000}ms")

print("g use time")
real_time = 0
for g in [use_time_g1, use_time_g2, use_time_g3]:
    real_time += sum(g)
    print(g)

print(f"real time: {real_time}ms")

print("timeline")
for timeline in [timelines_f1, timelines_f2, timelines_f3]:
    print(timeline)

"""
结论: python伪多线程
"""
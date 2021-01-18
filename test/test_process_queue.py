from multiprocessing import Process, Queue
import multiprocessing as mp
import os, time, random
import logging

def g1(step):
    # global total_use_time
    # print("g1 start time", time.time() * 1000)
    # t1 = time.time()
    x = 1
    for i in range(200000):
        x *= i
    # total_use_time += time.time() - t1
    # print("g1 end time", time.time() * 1000)

def g2(step):
    # global total_use_time
    # print('g2 start time', time.time() * 1000)
    # t1 = time.time()
    x = 1
    for i in range(100000):
        x *= i
    # print("g2 end time", time.time() * 1000)
    # total_use_time += time.time() - t1

def g3(step):
    # global total_use_time
    # print('g3 start time', time.time() * 1000)
    # t1 = time.time()
    x = 1
    for i in range(400000):
        x *= i
    # print("g3 end time", time.time() * 1000)
    # total_use_time += time.time() - t1


# 写数据进程执行的代码:
def write(q, share_var, share_lock):
    # print("ID of process running write: {}".format(os.getpid())) 
    # print("begin write", (time.time() * 1000))
    for i in range(10):
        t1 = time.time()
        value = 'A' + str(i)
        g1(i)
        t2 = time.time() - t1
        print(f"{i}_fun1, {t2 * 1000}ms")
        share_lock.acquire()
        share_var.value += t2
        share_lock.release()
        # print("begin write", i, (time.time() * 1000))
        q.put(value)
        # print("end write", i, (time.time() * 1000))

# 读数据进程执行的代码:
def read(q, subq, share_var, share_lock):
    # print("ID of process running read: {}".format(os.getpid())) 
    # print("begin read", (time.time() * 1000))
    for i in range(10):
        # print("begin read", i, (time.time() * 1000))
        value = q.get()
        # print("read get", i, value, (time.time() * 1000))
        t1 = time.time()
        g2(i)
        t2 = time.time() - t1
        print(f"{i}_fun2, {t2 * 1000}ms")
        share_lock.acquire()
        share_var.value += t2
        share_lock.release()
        # print("start put", i, value, (time.time() * 1000))
        subq.put(value)
        # print("end put", i, value, (time.time() * 1000))
        # print('Put %s to subqueue...' % value)

def subread(q, share_var, share_lock):
    # print("ID of process running subread: {}".format(os.getpid())) 
    # print("begin subread", (time.time() * 1000))
    for i in range(10):
        # print("begin subread", i, (time.time() * 1000))
        value = q.get()
        # print("subread get", i, value, (time.time() * 1000))
        t1 = time.time()
        g3(i)
        t2 = time.time() - t1
        # print(f"{i}_fun3, {t2 * 1000}ms")
        share_lock.acquire()
        share_var.value += t2
        share_lock.release()

if __name__=='__main__':
    print("Main process: {}".format(os.getpid())) 
    share_var = multiprocessing.Manager().Value('f', 0)
    share_lock = multiprocessing.Manager().Lock()
    # 父进程创建Queue，并传给各个子进程：
    # print("use time", time.time() * 1000) # 488.90
    q = Queue(1)
    subq = Queue(1)
    pw = Process(target=write, args=(q, share_var, share_lock))
    pr = Process(target=read, args=(q,subq, share_var, share_lock))
    psr = Process(target=subread, args=(subq, share_var, share_lock))

    st = time.time()
    # print("start", st) # 501.98
    # 启动子进程pw，写入:
    pw.start()
    pr.start()
    psr.start()
    # 等待pw结束:
    pw.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    #
    print("write task end: ", (time.time() - st) * 1000)
    pr.join()
    print("read task end: ", (time.time() - st) * 1000)
    psr.join()
    print("subread task end: ", (time.time() - st) * 1000)

    print("real use time", share_var.value * 1000)
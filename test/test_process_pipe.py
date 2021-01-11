from multiprocessing import Process, Queue, Pipe
import os, time, random
import logging

use_time_g1 = []
use_time_g2 = []
use_time_g3 = []

def g1(step):
    print("g1 start time", time.time() * 1000)
    x = 1
    for i in range(200000):
        x *= i
    print("g1 end time", time.time() * 1000)

def g2(step):
    print('g2 start time', time.time() * 1000)
    x = 1
    for i in range(100000):
        x *= i
    print("g2 end time", time.time() * 1000)


def g3(step):
    print('g3 start time', time.time() * 1000)
    x = 1
    for i in range(400000):
        x *= i
    print("g3 end time", time.time() * 1000)

# 写数据进程执行的代码:
def write(p):
    # print("ID of process running write: {}".format(os.getpid())) 
    print("begin write", (time.time() * 1000))
    for i in range(10):
        value = 'A' + str(i)
        g1(i)
        print("begin write", i, (time.time() * 1000))
        p.send(i)
        print("end write", i, (time.time() * 1000))

# 读数据进程执行的代码:
def read(q, subq):
    # print("ID of process running read: {}".format(os.getpid())) 
    print("begin read", (time.time() * 1000))
    while True:
        try:
        print("begin read", i, (time.time() * 1000))
        value = q.get()
        print("read get", i, value, (time.time() * 1000))
        g2(i)
        print("start put", i, value, (time.time() * 1000))
        subq.put(value)
        print("end put", i, value, (time.time() * 1000))
        # print('Put %s to subqueue...' % value)

def subread(q):
    # print("ID of process running subread: {}".format(os.getpid())) 
    print("begin subread", (time.time() * 1000))
    for i in range(10):
        print("begin subread", i, (time.time() * 1000))
        value = q.get()
        print("subread get", i, value, (time.time() * 1000))
        g3(i)

if __name__=='__main__':
    print("Main process: {}".format(os.getpid())) 

    # 父进程创建Queue，并传给各个子进程：
    print("use time", time.time() * 1000) # 488.90
    wp_in, rp_out = Pipe()
    rp_in, srp_out = Pipe()
    pw = Process(target=write, args=(wp_in,))
    pr = Process(target=read, args=(rp_out, rp_in))
    psr = Process(target=subread, args=(srp_out,))

    st = time.time() * 1000
    print("start", st) # 501.98
    # 启动子进程pw，写入:
    pw.start()
    pr.start()
    psr.start()
    # 等待pw结束:
    pw.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    #
    print("write task end: ", time.time() * 1000)
    pr.join()
    print("read task end: ", time.time() * 1000)
    psr.join()
    print("subread task end: ", time.time() * 1000 - st)
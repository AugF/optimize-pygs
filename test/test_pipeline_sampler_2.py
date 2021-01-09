"""
sampler, cuda+training
"""
import time
from threading import Thread

pre_fun2, cur_fun2 = None, None

t1, t2 = 1, 4

def fun2(i):
    global pre_fun2, cur_fun2
    while pre_fun2 is not None and pre_fun2.is_alive():
        pass
    print(f"{i}_fun2 begin")
    pre_fun2 = cur_fun2
    time.sleep(t2)       
    print(f"{i}_fun2 end")

for i in range(1, 4):
    time.sleep(t1)
    print(f"{i}_fun1")
    cur_fun2 = Thread(target=fun2, args=[i])
    cur_fun2.start()

pre_fun2.join()
cur_fun2.join()    
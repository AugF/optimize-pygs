"""
sampler, cuda+training
"""
import time
from threading import Thread

pre_fun2, cur_fun2 = None, None
x = 0
# def fun1(i):
#     time.sleep(2)
    
def fun2(i):
    global pre_fun2, cur_fun2, x
    while pre_fun2 is not None and pre_fun2.is_alive():
        pass
    print(f"{i}_fun2 begin: {x}")
    pre_fun2 = cur_fun2
    time.sleep(3)
    x += i        
    print(f"{i}_fun2 end: {x}")

for i in range(1, 4):
    time.sleep(i)
    print(f"{i}_fun1")
    cur_fun2 = Thread(target=fun2, args=[i])
    cur_fun2.start()

pre_fun2.join()
cur_fun2.join()    
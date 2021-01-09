"""
sample + to + trainingpre 
"""
import time
import random
from threading import Thread

t1, t2, t3 = 2, 1, 4
pre_fun2, cur_fun2 = None, None
pre_fun3, cur_fun3 = None, None

x = 0 # 检测fun2是否按序执行
y = 0 # 检测fun3是否按序执行
threads = []

def fun3(s, i): # 负责 *2
    global pre_fun3, cur_fun3, y
    while pre_fun3 is not None and pre_fun3.is_alive(): # pre_fun3->cur_fun3
        pass
    print(f"{i}_fun3 begin")
    pre_fun3 = cur_fun3
    time.sleep(t3)
    y += 1
    print(f"{i}_fun3 end: rank={y}, value={s*2}")
    
def fun2(i): # 负责 +1
    global pre_fun2, cur_fun2, x
    while pre_fun2 is not None and pre_fun2.is_alive(): # pre_fun2->cur_fun2
        pass
    print(f"{i}_fun2 begin")
    pre_fun2 = cur_fun2
    time.sleep(t2)
    x += 1       
    print(f"{i}_fun2 end: rank={x}, value={i*2}")
    cur_fun3 = Thread(target=fun3, args=[i * 2, i]) # fun2->fun3
    threads.append(cur_fun3)
    cur_fun3.start()

st0 = time.time()
for i in range(4): # dataloader
    time.sleep(t1)
    print(f"{i}_fun1")
    cur_fun2 = Thread(target=fun2, args=[i]) # 保证fun1->fun2
    threads.append(cur_fun2)
    cur_fun2.start()

for t in threads:
    t.join()
st1 = time.time()
print(f"pipeline use time: {st1 - st0}s")

# for i in range(4):
#     time.sleep(t1)
#     time.sleep(t2)
#     time.sleep(t3)

# print(f"without pipeline use time: {time.time() - st1}s")
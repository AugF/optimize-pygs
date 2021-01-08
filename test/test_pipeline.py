import time
import random
from threading import Thread

t1, t2, t3 = 2, 1, 8
res = []

def g1():
    time.sleep(t1)

def g2():
    time.sleep(t2)

def g3():
    time.sleep(t3)

threads = None
def fun1(i):
    g1()
    print(f"{i}_fun1")
    return i

def fun3(i, t3=4):
    g3()
    res.append(i)
    print(res)
    print(f"{i}_fun3")

def fun2(i, t2=1):
    g2()
    print(f"{i}_fun2")
    task3 = Thread(target=fun3, args=[i])
    threads.append(task3)
    task3.start()

def test_pipeline():
    t0 = time.time()
    for num in range(3):
        res1 = fun1(num) # task1 
        task2 = Thread(target=fun2, args=[res1])
        threads.append(task2)
        task2.start()

    for t in threads:
        t.join()
    print(f"pipeline use time: {time.time() - t0}")

    t1 = time.time()
    for num in range(3):
        g1()
        g2()
        g3()
    print(f"without pipeline use time: {time.time() - t1}")

for i in range(1):
    # t1, t2, t3 = random.randint(1,4), random.randint(1,4), random.randint(1,4)
    threads = []
    test_pipeline()
import time

def fun1(id):
    t1 = time.time()
    x = 1
    for i in range(200000):
        x *= i
    t2 = time.time()
    print(id, "fun1 use time", t2 - t1)

def fun2(id):
    t1 = time.time()
    x = 1
    for i in range(100000):
        x *= i
    t2 = time.time()
    print(id, "fun2 use time", t2 - t1)

def fun3(id):
    t1 = time.time()
    x = 1
    for i in range(400000):
        x *= i
    t2 = time.time()
    print(id, "fun3 use time", t2 - t1)
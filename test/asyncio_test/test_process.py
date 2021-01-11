import time


t1 = time.time()
x = 1
for i in range(600000):
    x *= i
print("use time", (time.time() - t1) * 1000)
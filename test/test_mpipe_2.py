import time
import sys
from mpipe import OrderedStage, Pipeline

if len(sys.argv) >= 5:
    print("python file.py st1 st2 st3 total_num")
    st1, st2, st3, total_num = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
else:
    st1, st2, st3 = 0.02, 0.01, 0.04
    total_num = 10

print(f"st1={st1}, st2={st2}, st3={st3}, total_num={total_num}")
x = 1
def increment(value):
    time.sleep(st1)
    return value

def double(value):
    time.sleep(st2)
    return value

def echo(value):
    time.sleep(st3)
    print(value, time.time())
    return value

t1 = time.time()
stage1 = OrderedStage(increment)
stage2 = OrderedStage(double)
stage3 = OrderedStage(echo)
stage1.link(stage2)
stage2.link(stage3)
pipe = Pipeline(stage1)

for number in range(total_num):
    pipe.put(number)

pipe.put(None)

# for res in pipe.results():
#     print(res)
print("use time: ", time.time() - t1, "except time: ", sum([st1, st2, st3]) + max([st1, st2, st3]) * (total_num - 1), "original time:", total_num * (st1 + st2 + st3))

t1 = time.time()
for i in range(total_num):
    echo(double(increment(i)))
print("origin time", time.time() - t1)
    

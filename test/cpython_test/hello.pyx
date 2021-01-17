# https://cython.readthedocs.io/en/latest/src/quickstart/build.html
from tools import fun1
from cython.parallel import prange, parallel
from libc.stdio cimport printf
import time

t1 = time.time()
for i in prange(2, num_threads=2):
    with gil:
        fun1(i)

t2 = time.time()
for i in range(2):
    fun1(i)

t3 = time.time()
print("use time", t2 - t1, t3 - t2)
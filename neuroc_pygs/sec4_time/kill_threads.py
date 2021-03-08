import os
import sys

pyPath = '-u exp2_sampling_inference.py'

lines = os.popen('ps -aux | grep %s' % pyPath)
for path in lines:
    progress =  path.split(' ')[1]
    client   =  path.split(' ')[6].split('/')[0]
    if client=="pts":
        continue

    print(progress , client)
    os.popen('kill -9 %s' % progress)
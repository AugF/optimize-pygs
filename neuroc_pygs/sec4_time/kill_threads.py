import os
import sys

pyPath = ['/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/base_epoch.py',
     '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/opt']

for pyp in pyPath:
    lines = os.popen('ps -aux | grep %s' % pyp)
    for path in lines:
        progress =  path.split(' ')[1]
        client   =  path.split(' ')[6].split('/')[0]
        if client=="pts":
            continue

        print(progress , client)
        os.popen('kill -9 %s' % progress)
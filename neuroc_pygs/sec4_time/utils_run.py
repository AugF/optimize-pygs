import os
import sys


def kill_threads():
    pyPath = ['/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/epoch_base_sampling.py',
     '/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/opt']

    for pyp in pyPath:
        lines = os.popen('ps -aux | grep %s' % pyp)
        for path in lines:
            progress =  path.split(' ')[1]
            client   =  path.split(' ')[6].split('/')[0]
            if client == "pts":
                continue

            print(progress , client)
            os.popen('kill -9 %s' % progress)
     
       
def run_pics():
    for file in os.listdir('.'):
        if 'pics_thesis' in file or 'ppt' in file:
            print(file)
            os.system('python ' + file)
    

if __name__ == '__main__':
    run_pics()
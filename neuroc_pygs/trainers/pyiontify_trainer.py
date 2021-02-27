#https://www.cnblogs.com/houjun/p/10407423.html
import time
import os
import subprocess

from neuroc_pygs.configs import PROJECT_PATH

# 有效果，甚至很好，丰富测试
def opt_trainer():
    print("start")
    t1 = time.time()
    pro_train = subprocess.Popen("python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/trainers/pyiontify_train.py --epochs 10", shell=True)
    pro_eval = subprocess.Popen("python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/trainers/pyiontify_eval.py --epochs 10", shell=True)
    print(f'Pid train process: {pro_train.pid}, eval_process: {pro_eval.pid}')
    pro_eval.communicate()
    pro_eval.wait()
    pro_train.kill()
    t2 = time.time()
    print(f'end use time: {t2 - t1}s')


if __name__ == '__main__':
    opt_trainer()

    
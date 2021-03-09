import time, os
import subprocess
from neuroc_pygs.configs import PROJECT_PATH

headers = ['Name', 'Baseline', 'Batch Opt', 'Epoch Opt', 'Opt']

def opt_epoch(args=''):
    pro_train = subprocess.Popen(
        "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/opt_full_train.py " + args, shell=True)
    pro_eval = subprocess.Popen(
        "python /home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/opt_full_eval.py " + args, shell=True)
    print(f'Pid train process: {pro_train.pid}, eval_process: {pro_eval.pid}')
    pro_eval.communicate()
    pro_eval.wait()
    pro_train.kill()


def epoch(args=''):
    pro = subprocess.Popen("/home/wangzhaokang/wangyunpan/gnns-project/optimize-pygs/neuroc_pygs/sec4_time/base_full_epoch.py " + args, shell=True)
    pro.communicate()
    pro.wait()


if __name__ == '__main__':
    # opt_epoch()
    epoch()

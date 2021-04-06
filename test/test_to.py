from multiprocessing import Process
import os


def child_process_func(name):
    """
    当前函数是交给子进程实现
    :return:
    """
    print('当前执行的进程是子进程，进程编号是{},父进程编号是{},当前子进程的名字是{}'.format(os.getpid(),os.getppid(),name))


if __name__ == '__main__':
    # TODO 打印父进程ID
    print("父进程的ID", os.getpid())
    # TODO 创建子进程
    cp = Process(target=child_process_func,args=('子进程',)) # 元组只有一个元素时要加,target说明使用的方法/对象
    # def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):       //agrs=()，打包成元组
    cp.start()
    print(cp.name,cp.pid) # 获取子进程的名字，进程名为1  获取子进程的pid

    # print('结束主进程')    # 子进程没结束，主进程就结束了

    cp.join()               # 等所有进程完成后，主进程才结束
    print('结束主进程')

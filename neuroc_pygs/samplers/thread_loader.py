import time
import sys 
from threading import Thread
from queue import Empty, Full, Queue 
from neuroc_pygs.samplers.thread_utils import thread_killer, sampler_threads, cuda_threads


class ThreadLoader(object):
    def __init__(self, loader, sampler, device, data=None):
        self.sampler_queue, self.cuda_queue = Queue(maxsize=1), Queue(maxsize=1)
        self.sampler_killer = thread_killer()
        self.sampler_killer.set_tokill(False)
        self.len = len(loader)
        
        for _ in range(1): # preprocess
            t = Thread(target=sampler_threads, args=(self.sampler_killer, self.sampler_queue, loader))
            t.start()

        self.cuda_killer = thread_killer()
        self.cuda_killer.set_tokill(False)
        cuda_thread = Thread(target=cuda_threads, args=(self.cuda_killer, self.cuda_queue, self.sampler_queue, sampler, device, data))
        cuda_thread.start()
        # need to wait
        
    def __next__(self):
        return self.cuda_queue.get(block=True)
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.len
        
    def next(self):
        return self.__next__()
    
    def stop(self):
        self.sampler_killer.set_tokill(True)
        self.cuda_killer.set_tokill(True)
        for _ in range(4):
            try:
                self.sampler_queue.get(block=True, timeout=1)
                self.cuda_queue.get(block=True, timeout=1)
            except Empty:
                pass



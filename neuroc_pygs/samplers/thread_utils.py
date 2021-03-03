from neuroc_pygs.utils import to

class thread_killer:
    def __init__(self):
        self.to_kill = False 
        
    def __call__(self):
        return self.to_kill
    
    def set_tokill(self, tokill):
        self.to_kill = tokill


def sampler_threads(tokill, sampler_queue, loader):
    while tokill() == False:
        for batch in loader:
            sampler_queue.put(batch, block=True)
        if tokill() == True: 
            return 


def cuda_threads(tokill, cuda_queue, sampler_queue, sampler, device, data=None):
    while tokill() == False:
        flag = False
        batch = sampler_queue.get(block=True)
        # task2 start
        if sampler == 'graphsage':
            batch_size, n_id, adjs = batch
            adjs = [adj.to(device=device, non_blocking=flag) for adj in adjs]
            x, y = data.x[n_id], data.y[n_id[:batch_size]]
            x, y = x.cuda(device, non_blocking=flag), y.cuda(device, non_blocking=flag)
            batch = [batch_size, x, y, adjs]
        else:
            batch = batch.to(device, non_blocking=flag)
        # task2 end
        cuda_queue.put(batch)
        if tokill() == True: 
            return 
        
import threading
import queue as Queue

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()
        self.exhausted = False

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        if self.exhausted:
            raise StopIteration
        else:
            next_item = self.queue.get()
            if next_item is None:
                raise StopIteration
            return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

#decorator
class background:
    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch
    def __call__(self, gen):
        def bg_generator(*args,**kwargs):
            return BackgroundGenerator(gen(*args,**kwargs), max_prefetch=self.max_prefetch)
        return bg_generator


if __name__ == '__main__':
    import time
    from neuroc_pygs.options import build_train_loader, get_args, build_dataset
    args = get_args()
    data = build_dataset(args)
    train_loader = build_train_loader(args, data)

    iter1 = iter(train_loader)
    iter2 = BackgroundGenerator(iter(train_loader))
    t1 = time.time()
    for _ in iter1: pass
    t2 = time.time()
    for _ in iter2: pass
    t3 = time.time()
    print(f'use time: {t2 - t1}, opt time: {t3 - t2}')

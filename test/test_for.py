class A:
    num = 10
    def __init__(self):
        self.cnt = 0
    
    def __iter__(self):
        print("yes")
        return self
    
    def __next__(self):
        self.cnt += 1
        if self.cnt >= A.num:
            self.cnt = 0
            raise StopIteration
        return self.cnt   

a = A()
for i in a:
    print(i)     
    
a_iter = iter(a)
for i in range(A.num):
    print(next(a_iter))
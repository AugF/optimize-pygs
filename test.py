import datetime
from threading import Timer

def tt():
    print(datetime.datetime.now())

if __name__ == '__main__':
    n=datetime.datetime.now() + datetime.timedelta(seconds=10) #execute after 30s
    for i in range(10):
        t=Timer((n-datetime.datetime.
now()).seconds,tt)
        t.start()
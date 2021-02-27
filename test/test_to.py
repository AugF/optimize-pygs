import pyinotify
import asyncio


class EventHandler(pyinotify.ProcessEvent):
    def process_IN_CREATE(self, event):
        if not event.dir:
            print("Got new file: ", event.pathname)

wm = pyinotify.WatchManager()  # Watch Manager
mask = pyinotify.IN_DELETE | pyinotify.IN_CREATE  # watched events

loop = asyncio.get_event_loop()

notifier = pyinotify.AsyncioNotifier(wm, loop, default_proc_fun=EventHandler())
wdd = wm.add_watch('.', mask, rec=True, auto_add=True)

try:
    loop.run_forever()
except:
    print('\nshutting down...')

loop.stop()
notifier.stop()
import os

for file in os.listdir('.'):
    if 'pics_thesis' in file and 'ppt' in file:
        print(file)
        os.system('python ' + file)
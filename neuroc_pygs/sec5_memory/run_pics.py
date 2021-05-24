import os

for file in os.listdir('.'):
    if 'pics' == file[:4]:
        print(file)
        os.system('python ' + file)
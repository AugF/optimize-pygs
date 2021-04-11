import os

for file in os.listdir('.'):
    if 'thesis' in file:
        print(file)
        os.system('python ' + file)
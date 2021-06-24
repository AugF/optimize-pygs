import os

for file in os.listdir('./'):
    if 'pics' in file:
        print(file)
        os.system(f'python {file}')
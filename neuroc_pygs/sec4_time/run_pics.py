import os

for file in os.listdir('.'):
    if 'pics_epoch_sampling' in file:
        os.system('python ' + file)
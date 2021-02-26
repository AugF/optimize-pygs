with open('test.keys', 'a+') as f:
    x = f.read().strip().split('\n')
    print(x)

f = open('test.keys', 'a+')
for i in range(10):
    f.write(str(i) + '\n')
f.close()
with open('test.keys') as f:
    x = f.read().strip().split('\n')
    print(x)
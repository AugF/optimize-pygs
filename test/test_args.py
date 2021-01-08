

def fun(arg1, **args):
    assert 'name' in args
    print(args['name'])
    
    
fun(1, name=1)
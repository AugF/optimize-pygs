import commands
 
status = commands.getstatus('cat /etc/passwd')
print(status)
output = commands.getoutput('cat /etc/passwd')
print(output)
(status, output) = commands.getstatusoutput('cat /etc/passwd')
print(status, output)
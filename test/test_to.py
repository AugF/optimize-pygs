import torch
import time

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
# Initialise cuda tensors here. E.g.:
A = torch.rand(10000, 10000, device = 'cuda')
# B = torch.rand(10000, 10000, device = 'cuda')
# Wait for the above tensors to initialise.
print(dir(A))
A.to()
# torch.cuda.synchronize()
# t0 = time.time()
# with torch.cuda.stream(s1):
#     C = torch.mm(A, A)
# with torch.cuda.stream(s2):
#     D = torch.mm(B, B)
# # Wait for C and D to be computed.
# torch.cuda.synchronize()
# torch.to()

# t1 = time.time()
# C = torch.mm(A, A)
# D = torch.mm(B, B)
# torch.cuda.synchronize()
# t2 = time.time()
# print("use time", t1 - t0, t2 - t1)
#?
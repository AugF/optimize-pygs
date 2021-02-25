import torch

tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
tensor.to(torch.float64)

cuda0 = torch.device('cuda:0')
tensor.cuda(cuda0)

# tensor.to(cuda0, dtype=torch.float64)
# print(tensor, tensor.device)
# other = torch.randn((), dtype=torch.float64, device=cuda0)
# tensor.to(other, non_blocking=True)
# print(tensor, tensor.device)
# print(other, other.device)

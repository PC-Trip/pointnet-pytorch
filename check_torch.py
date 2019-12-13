import torch


x = torch.rand(5, 3)
print('torch.rand(5, 3): {}')
print(x)
print('torch.cuda.is_available(): {}'.format(torch.cuda.is_available()))


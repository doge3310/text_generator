import torch

lst = torch.randint(1, 4100, (59, 60))

print(lst[: 2].type())

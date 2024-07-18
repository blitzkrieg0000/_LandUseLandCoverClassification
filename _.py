import torch

batch_size = 1
segments = 1

segments = [len(i) for i in torch.chunk(torch.range(0, batch_size), segments)]
print(segments)

import torch

loss = torch.nn.CrossEntropyLoss()



input = torch.randn(1, 2, 3, 5, requires_grad=True)
target = torch.empty(1, 3, 5, dtype=torch.long).random_(2)


print(input, "\n", target)

output = loss(input, target)
output.backward()
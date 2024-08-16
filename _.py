import torch

classes = torch.tensor(
    [
        [
            [1.0, 50.0, 100.0],
            [1.0, 50.0, 100.0],
            [1.0, 50.0, 100.0]
        ],
        [
            [1.0, 50.0, 100.0],
            [1.0, 50.0, 100.0],
            [1.0, 50.0, 100.0]
        ]
    ]
)


# print(classes.shape)
# pred = torch.nn.functional.softmax(classes)
# print(pred)


print(classes.sum(dim=(1)))

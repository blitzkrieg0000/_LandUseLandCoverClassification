#%%
import torch
#                0   1   2    3
t = torch.randn(100, 4, 512, 512)
print(t.shape)

unfolded = t.unfold(2, 16, 16)
print(unfolded.shape)

unfolded = unfolded.unfold(3, 16, 16)
print(unfolded.shape)


# 100, 4, 32, 32, 16, 16









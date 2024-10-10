import torch

DEVICE = torch.device(torch.cuda._get_device(0) if torch.cuda.is_available() else "cpu")

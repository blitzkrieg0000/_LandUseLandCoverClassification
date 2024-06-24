from torch.utils.data import Dataset


class BaseDatasetProcessor(Dataset):
	def __init__(self):
		super().__init__()
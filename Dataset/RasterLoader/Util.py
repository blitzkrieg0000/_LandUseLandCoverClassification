# =================================================================================================================== #
#! FUNCTION
# =================================================================================================================== #
import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb
import torch
from Tool.Core import ChangeMaskOrder, ConsoleLog, LogColorDefaults
from rastervision.pytorch_learner import SemanticSegmentationVisualizer


def DataChunkRepeatCounts(batch_size, batch_data_chunk_number):
	"""[1 1 1 1 1 1 1 1], [2 2 2 2], [4 4], [3 3 2], [8]"""
	chunkSize = torch.clamp(torch.tensor(batch_data_chunk_number), 1, batch_size)
	chunks = torch.chunk(torch.arange(batch_size), chunkSize) 
	return list(map(len, chunks))


def WorkerInitFN(worker_id):
	worker_info = torch.utils.data.get_worker_info()
	dataset = worker_info.dataset
	dataset.set_worker_info(worker_id, worker_info.num_workers)


def CollateFN(batch):
	# TODO Eksik olan None değerler, tekrar burada verisetinden çekilebiliyor mu? Dene (Recursive gibi olabilir).
	if batch is None:
		return None, None
	data, label = zip(*batch)
	return torch.stack([d for d in data if d is not None]), torch.stack([l for l in label if l is not None])


def ShowDatasetViaVisualizer(dataset):
	vis = SemanticSegmentationVisualizer(class_names=["background", "foreground"], class_colors=["black", "white"])
	x, y = vis.get_batch(dataset, 4)
	vis.plot_batch(x, y, show=True)


def ShowBatchRaster(raster):
	fig, ax = plt.subplots(4, 4, figsize=(7, 7))
	for i in range(12):
		ax[i%4, i//4].matshow(raster[:, :, [i]], cmap="viridis")
		ax[i%4, i//4].axis("off")

	plt.show()


def VisualizeData(dataloader, limit=None):
	# print("\nDataloader Size:", len(DATALOADER))
	
	fig, axs = plt.subplots(4, 4, figsize=(12, 12))

	for i, (buffer, mask) in enumerate(dataloader):
		print(i, buffer.shape, mask.shape, "\n-----------------\n" )
		for bn in range(buffer.shape[0]):
			for i in range(16):
				axs[i%4, i//4].axis("off")
				if i<buffer.shape[1]:
					axs[i%4, i//4].imshow(buffer[bn, i].numpy(), cmap="viridis")
					axs[i%4, i//4].set_title(f"Band {i+1}")
			print("Labels: ", mask[bn].unique())
			axs[3, 3].imshow(mask[bn])
			axs[3, 3].set_title("Ground Truth")
			plt.pause(1)
		
		if limit is not None and i >= limit:
			break

	plt.tight_layout()
	plt.show()


def VisualizeClassDistribution(dataloader, num_classes=9):
	fig, ax = plt.subplots()
	plt.tight_layout()

	uLabels = set()
	hist = np.zeros(num_classes)
	colors = sb.color_palette("husl", len(hist))
	classes = torch.arange(1, num_classes + 1)

	ConsoleLog(f"Main Process Id: {os.getpid()}", LogColorDefaults.Warning)
	for i, (inputs, targets) in enumerate(dataloader):
		# inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
		print(f"Step: {i}", inputs.shape, targets.shape)

		targets = ChangeMaskOrder(targets, classes)

		uniqueLabels: torch.Tensor = targets.unique()
		uLabels.update(set(uniqueLabels.tolist()))
		for label in uniqueLabels:
			hist[label] += 1
		
		print("Labels: ", uLabels)
		
		ax.bar(np.arange(num_classes), hist, color=colors)
		ax.set_xticks(np.arange(num_classes))
		plt.pause(0.1)

	plt.show()


def VisualizePrediction(buffer, mask, predicted):
	# print("\nDataloader Size:", len(DATALOADER))
	
	fig, axs = plt.subplots(4, 4, figsize=(12, 12))

	for bn in range(buffer.shape[0]):
		for i in range(16):
			axs[i%4, i//4].axis("off")
			if i<buffer.shape[1]:
				image = buffer[bn, i].cpu().numpy()
				axs[i%4, i//4].imshow(image, cmap="viridis")
				axs[i%4, i//4].set_title(f"Band {i+1}")
		
		axs[2, 3].imshow(mask[bn].cpu().numpy(), cmap="viridis")
		axs[2, 3].set_title("Ground Truth")
		axs[3, 3].imshow(predicted[bn].cpu().numpy(), cmap="viridis")
		axs[3, 3].text(0, 0, "Predicted", fontsize=12, color="blue", weight="bold")

	plt.tight_layout()
	plt.show()

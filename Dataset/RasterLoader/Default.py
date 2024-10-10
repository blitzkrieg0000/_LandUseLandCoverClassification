from enum import Enum
from typing import Annotated, Any, List, Tuple

from pydantic import BaseModel
from rastervision.core.data import ClassConfig

from Dataset.RasterLoader.Util import DataChunkRepeatCounts


class SegmentationDatasetConfig(BaseModel):
	"""Segmentation Veri Seti Konfigürasyon Sınıfı"""
	ClassNames: Annotated[List[str], "Class Names"]
	ClassColors: Annotated[List, "Class Colors"]
	NullClass: Annotated[Any, "Null Class"] = None
	MaxWindowsPerScene: Annotated[int | float | None, "Max Windows Per Scene"]
	DatasetRootPath: Annotated[str, "Dataset Index File"]
	PatchSize: Annotated[Tuple[int, int] | int, "Patch Size"]
	PaddingSize: Annotated[int, "Padding Size"]
	Shuffle: Annotated[bool, "Shuffle"]  
	RandomLimit: Annotated[int, "RandomLimit"] = None
	RandomPatch: Annotated[bool, "Random Patch"] = True
	BatchDataChunkNumber: Annotated[int, "Batch Dataset Chunk Number"] = 1
	BatchSize: Annotated[int, "Batch Size"] = 1
	DropLastBatch: Annotated[bool, "Drop Last Batch"] = True
	StrideSize: Annotated[int, "Sliding Window Stride Size"] = 112
	ChannelOrder: Annotated[List[int], "Channel Order"] = None
	DataFilter: Annotated[List[str], "Data Filter By File Name Regex"] = None
	DataLoadLimit: Annotated[int, "Data Limiter"] = None
	Verbose: Annotated[bool, "Verbose"] = False

	@property
	def BatchRepeatDataSegment(self):
		return DataChunkRepeatCounts(self.BatchSize, self.BatchDataChunkNumber)
	
	@property
	def IteratorMargin(self):
		return max(self.BatchRepeatDataSegment)
	
	@property
	def FixedPatchSize(self):
		if isinstance(self.PatchSize, Tuple):
			patchX, patchY = self.PatchSize[0], self.PatchSize[1]
		else:
			patchX, patchY = self.PatchSize, self.PatchSize

		return (patchX, patchY)

	@property
	def DefaultLabelSourceConfig(self):
		return ClassConfig(names=self.ClassNames, colors=self.ClassColors, null_class=self.NullClass)



class DataReadType(Enum):
	IndexMetaFile = 0


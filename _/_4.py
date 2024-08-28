from functools import cached_property
from typing import Annotated, List, Tuple

from pydantic import BaseModel
import torch


def DataChunkRepeatCounts(batch_size, batch_data_chunk_number):
    """[1 1 1 1 1 1 1 1], [2 2 2 2], [4 4], [3 3 2], [8]"""
    chunkSize = torch.clamp(torch.tensor(batch_data_chunk_number), 1, batch_size)
    chunks = torch.chunk(torch.arange(batch_size), chunkSize) 
    return list(map(len, chunks))


# =================================================================================================================== #
#! CLASS
# =================================================================================================================== #
class SegmentationDatasetConfig(BaseModel):
    BatchDataChunkNumber: Annotated[int, "Batch Dataset Chunk Number"]=2
    BatchSize: Annotated[int, "Batch Size"]=8

    @property
    def BatchRepeatDataSegment(self):
        return DataChunkRepeatCounts(self.BatchSize, self.BatchDataChunkNumber)
    


config = SegmentationDatasetConfig()
print(config.BatchRepeatDataSegment)

print(-1%1)
from Dataset.Core import RemoteSensingDatasetManager
from Dataset.Enum import DatasetType
from Model.Enum import ModelType
from Train.Core import TrainManager


##! --------------- Model --------------- !##
manager = TrainManager(model_type=ModelType.UNET_3D)


##! --------------- Dataset --------------- !##
dataset = RemoteSensingDatasetManager().GetDataloader(DatasetType.Cukurova_IO_LULC)


##! --------------- Training --------------- !##
manager.Train(dataset)
del manager
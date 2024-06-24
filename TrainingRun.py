from Dataset.Core import RemoteSensingDatasetManager
from Dataset.Enum import DatasetType
from Model.Enum import ModelType
from Train.Core import TrainManager

wandb_configs = {
    "project" : "LULC_Project01",
    "wandb_entity" : "burakhansamli0-0-0-0",
    "group" : "Train_CustomUnet",
    "tags" : ["Cukurova_IO_LULC", "CustomUnet", "Train"],
    "consts" : {
        "architecture": "Unet",
        "dataset": "Cukurova_IO_LULC"
    }
}

train_config = {
    "EPOCH" : 50000,
    "BATCH_SIZE" : 16,
    "PATCH_SIZE" : 64
}

##! --------------- Model --------------- !##
manager = TrainManager(model_type=ModelType.UNET_3D, override_train_configs=train_config, wandb_kwargs=wandb_configs)


##! --------------- Dataset --------------- !##
dataset = RemoteSensingDatasetManager.GetDataloader(DatasetType.Cukurova_IO_LULC)


##! --------------- Training --------------- !##
manager.Train(dataset)
del manager
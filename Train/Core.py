import torch
import wandb

from Dataset.Core import RemoteSensingDatasetManager
from Dataset.DataProcess import DATA_TRANSFORMS_BY_MODEL, TRANSFORM_IMAGE
from Dataset.Enum import DatasetType
from Model.Core import ModelManager
from Model.Enum import ModelType
from Tool import ChangeMaskOrder
from Train.Const import TRAIN_DEFAULTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class WandbLogger():
    def __init__(self) -> None:
        self._APIKeyPath = "./data/asset/config/wandb/.apikey"
        self.InitializeWandb()
        self.__instance=None

    def ReadWandbKeyFromFile(self) -> str:
        with open(self._APIKeyPath, "r") as f:
            return f.read()
    
    def InitializeWandb(self, project, wandb_entity, group="Train_", save_dir="./data/", tags=["Train"], consts={}):
        """ Weights & Biases """
        if self.__instance is not None:
            raise ValueError("Wandb oturumu zaten açıldı.")
        
        wandb_key = self.ReadWandbKeyFromFile()
        wandb.login(key=wandb_key)
        self.__instance = wandb.init(
            project=project,       # Wandb Project Name
            entity=wandb_entity,   # Wandb Entity Name
            dir=save_dir,
            config=consts,
            group=group,
            tags=tags
        )
        return self.__instance
    
    def Log(self, verbose=True, **kwargs):
        try: 
            self.__instance.log(kwargs)
        except Exception as e: 
            if verbose: 
                print(e)

    def Save(self, path):
        self.__instance.save(path)

    def Teardown(self):
        self.__instance.finish()


class TrainManager():
    def __init__(self, model_type:ModelType=ModelType.UNET_2D) -> None:
        self.__WBLogger = None
        self.__WBSavePath = ".data/wandb/weight/custom02_unet.pth"
        self.ModelSavePath = "./weight/test.pth"
        self._ModelType=model_type
        self.__InitializeWandb()
    
    def __del__(self):
        self.__WBLogger.Teardown()


    def __InitializeWandb(self):
        self.__WBLogger = WandbLogger()
        self.__WBLogger.InitializeWandb(
            "LULC_Project01",
            "burakhansamli0-0-0-0",
            group="Train_CustomUnet",
            tags=["Cukurova_IO_LULC", "CustomUnet", "Train"],
            consts=TRAIN_DEFAULTS

        )

    def InitializeTrainParameters(self, **params):
        self.TrainParameters = params
    

    def PreprocessInput(self, inputs):
        _transform = DATA_TRANSFORMS_BY_MODEL[self._ModelType]["input_transform"]
        return _transform(inputs)

    def PreprocessTarget(self, targets):
        _transform = DATA_TRANSFORMS_BY_MODEL[self._ModelType]["target_transform"]
        return _transform(targets)

    def SaveModel(self, MODEL):
        torch.save(MODEL.state_dict(), self.ModelSavePath)
    

    def CalculateAccuracy(self, prediction, target, BATCH_SIZE, PATCH_SIZE):
        class_indices = torch.argmax(prediction, dim=1)  #TODO Unet3D dim2
        class_indices = class_indices.squeeze(1)
        accuracy = 100*((class_indices.flatten() == target.flatten()).sum() / PATCH_SIZE**2 / BATCH_SIZE)
        return accuracy.item()


    def Train(self, dataloader, verbose=True):
        ##! --------------- Model --------------- !##
        MODEL, criterion, optimizer = ModelManager().Create(self._ModelType)
        self.TrainParameters = TRAIN_DEFAULTS[self._ModelType]
        if verbose: print(MODEL)
        
        ##! --------------- Params --------------- !##
        EPOCH = self.TrainParameters.get("EPOCH", 1000)
        BATCH_SIZE = self.TrainParameters.get("BATCH_SIZE", 16)
        PATCH_SIZE = self.TrainParameters.get("PATCH_SIZE", 64)

        totalStep=0
        for epoch in range(EPOCH):
            totalAccuracy = 0
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                ##! --------------- Zeroing Optimizer's Gradient --------------- !##
                optimizer.zero_grad()
                
                ##! --------------- Preprocess --------------- !##
                inputs = self.PreprocessInput(inputs)
                targets = self.PreprocessTarget(targets)

                ##! --------------- Forward --------------- !##
                outputs = MODEL(inputs)
                
                ##! --------------- Loss --------------- !##
                loss = criterion(outputs, targets)

                ##! --------------- Backward --------------- !##
                loss.backward()
                optimizer.step()

                ##! --------------- Metrics --------------- !##
                accuracy = self.CalculateAccuracy(outputs, targets, BATCH_SIZE, PATCH_SIZE)
                totalAccuracy += accuracy
                totalStep += 1

                ###! --------------- Log --------------- !##
                if verbose: 
                    print(f"Epoch {epoch+1}/{EPOCH}, Train Loss: {loss.item()/BATCH_SIZE}, Train Accuracy: {accuracy}")

                self.__WBLogger.Log({
                    "epoch": epoch+1,
                    "train_loss": loss.item()/BATCH_SIZE,
                    "train_accuracy": accuracy
                })


        print(f"Epoch {epoch+1}/{EPOCH}, Average Accuracy: {totalAccuracy/totalStep}")
        self.__WBLogger.Save(self.__WBSavePath)



import torch
import wandb

from Dataset.DataTransform import DATA_TRANSFORMS_BY_MODEL
from Model.Core import ModelManager
from Model.Enum import ModelType
from Train.Const import TRAIN_DEFAULTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class WandbLogger():
    def __init__(self, **configs) -> None:
        self._APIKeyPath = "./data/asset/config/wandb/.apikey"
        self.__instance=None
        self.InitializeWandb(**configs)

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
    def __init__(self, model_type:ModelType=ModelType.UNET_2D, override_train_configs={}, wandb_kwargs={}, wb_logging=True) -> None:
        self.__WBLogger = None
        self.__WBSavePath = "./data/wandb/weight/custom02_unet.pth"
        self.ModelSavePath = "./weight/test.pth"
        self._ModelType=model_type
        self.TrainParameters = {}
        self.WBConfigs = wandb_kwargs
        self.WBLogging = wb_logging
        self.__InitializeTrainParameters(**override_train_configs)
    
    def __del__(self):
        if self.__WBLogger is not None:
            self.__WBLogger.Teardown()


    def __InitializeWandb(self, **wandb_kwargs):
        self.__WBLogger = WandbLogger(**wandb_kwargs)


    def __InitializeTrainParameters(self, **params):
        self.TrainParameters = TRAIN_DEFAULTS.get(self._ModelType, {})
        self.TrainParameters.update(params)
        if self.WBLogging:
            consts = self.WBConfigs.get("consts", {})
            consts.update({"learning_rate": self.TrainParameters["LEARNING_RATE"]})
            consts.update({"consts": self.TrainParameters["EPOCH"]})
            self.WBConfigs["consts"] = consts
            self.__InitializeWandb(**self.WBConfigs)
    

    def PreprocessInput(self, inputs):
        _transform = DATA_TRANSFORMS_BY_MODEL[self._ModelType]["input_transform"]
        return _transform(inputs)


    def PreprocessTarget(self, targets):
        _transform = DATA_TRANSFORMS_BY_MODEL[self._ModelType]["target_transform"]
        return _transform(targets)


    def PostprocessOutput(self, outputs):
        _transform = DATA_TRANSFORMS_BY_MODEL[self._ModelType]["output_transform"]
        return _transform(outputs)

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
        
        if verbose: 
            print(MODEL)
        
        ##! --------------- Params --------------- !##
        EPOCH = self.TrainParameters.get("EPOCH", 1000)
        BATCH_SIZE = self.TrainParameters.get("BATCH_SIZE", 16)
        PATCH_SIZE = self.TrainParameters.get("PATCH_SIZE", 64)

        totalStep = 0
        for epoch in range(EPOCH):
            totalAccuracy = 0
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                ##! --------------- Zeroing Optimizer's Gradient --------------- !##
                optimizer.zero_grad()
                
                ## --------------- Preprocess --------------- !##
                inputs = self.PreprocessInput(inputs)
                targets = self.PreprocessTarget(targets)

                ##! --------------- Forward --------------- !##
                outputs = MODEL(inputs)
                
                ##! --------------- Loss --------------- !##
                loss = criterion(outputs, targets)

                ##! --------------- Backward --------------- !##
                loss.backward()
                optimizer.step()

                ## --------------- Postprocess --------------- !##
                outputs = self.PostprocessOutput(outputs)
                
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



import torch
import wandb

from Dataset.Core import RemoteSensingDatasetManager
from Dataset.DataProcess import TRANSFORM_IMAGE
from Dataset.Enum import DatasetType
from Model.Core import ModelManager
from Model.Enum import ModelType
from Tool import ChangeMaskOrder
from Train.Config import TRAIN_DEFAULTS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainManager():
    def __init__(self, model_type:ModelType=ModelType.UNET_2D) -> None:
        self._ModelType=model_type
        self.ModelSavePath = "./weight/test.pth"


    def InitializeTrainParameters(self, **params):
        self.TrainParameters = params
    

    def PreprocessInput(self, inputs):
        return TRANSFORM_IMAGE(inputs)


    def SaveModel(self, MODEL):
        torch.save(MODEL.state_dict(), self.ModelSavePath)
    

    def CalculateAccuracy(self, prediction, target, BATCH_SIZE, PATCH_SIZE):
        class_indices = torch.argmax(prediction, dim=1)  #TODO Unet3D dim2
        class_indices = class_indices.squeeze(1)
        accuracy = 100*((class_indices.flatten() == target.flatten()).sum() / PATCH_SIZE**2 / BATCH_SIZE)
        return accuracy.item()


    def Train(self, dataset_type:DatasetType=DatasetType.Cukurova_IO_LULC, verbose=True):
        ##! --------------- Model --------------- !##
        MODEL, criterion, optimizer = ModelManager().Create(self._ModelType)
        self.TrainParameters = TRAIN_DEFAULTS[self._ModelType]
        if verbose: print(MODEL)
        
        ##! --------------- Dataset --------------- !##
        DATALOADER = RemoteSensingDatasetManager().GetDataloader(dataset_type)
        
        ##! --------------- Params --------------- !##
        EPOCH = self.TrainParameters.get("EPOCH", 1000)
        BATCH_SIZE = self.TrainParameters.get("BATCH_SIZE", 16)
        PATCH_SIZE = self.TrainParameters.get("PATCH_SIZE", 64)

        totalStep=0
        for epoch in range(EPOCH):
            totalAccuracy = 0
            for inputs, targets in DATALOADER:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                optimizer.zero_grad()
                
                #TODO inputs = inputs[:, 0:10, :, :]     #! Unet3D
                #TODO inputs = inputs.unsqueeze(1) #! Unet3D
                
                inputs = self.PreprocessInput(inputs)
                outputs = MODEL(inputs)
                
                #TODO one_hot_mask = one_hot_mask.unsqueeze(1) #! Unet3D

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # Accuracy
                accuracy = self.CalculateAccuracy(outputs, targets, BATCH_SIZE, PATCH_SIZE)
                totalAccuracy += accuracy
                totalStep += 1

                if verbose: print(f"Epoch {epoch+1}/{EPOCH}, Train Loss: {loss.item()/BATCH_SIZE}, Train Accuracy: {accuracy}")

                try:
                    wandb.log({
                        "epoch": epoch+1,
                        "train_loss": loss.item()/BATCH_SIZE,
                        "train_accuracy": accuracy
                    })
                except Exception as e:
                    if verbose:
                        print(e)
                
        print(f"Epoch {epoch+1}/{EPOCH}, Average Accuracy: {totalAccuracy/totalStep}")
        
        # wandb.save(".data/wandb/weight/custom02_unet.pth")
        # wandb.finish()


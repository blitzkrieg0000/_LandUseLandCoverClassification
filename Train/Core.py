import torch

from Dataset.Core import RemoteSensingDatasetManager
from Dataset.Enum import DatasetType
from Model.Core import ModelManager
from Model.Enum import ModelType
from Tool import ChangeMaskOrder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class TrainManager():
    def __init__(self, model_type:ModelType=ModelType.UNET_2D) -> None:
        self._ModelType=model_type
        self.TrainParameters = {}

    def InitializeTrainParameters(self, **params):
        self.TrainParameters = params
    
    def Preprocess(self):
        ...

    def Train(self, dataset_type:DatasetType=DatasetType.Cukurova_IO_LULC, verbose=True):
        ##! --------------- Model --------------- !##
        MODEL, criterion, optimizer = ModelManager().Create(self._ModelType)
        if verbose:
            print(MODEL)
        
        ##! --------------- Dataset --------------- !##
        DATALOADER = RemoteSensingDatasetManager().GetDataloader(dataset_type)
        
        
        ##! --------------- Params --------------- !##
        classes = torch.tensor([1, 2, 4, 5, 7, 8, 9, 10, 11])
        EPOCH = self.TrainParameters.get("EPOCH", 1000)
        BATCH_SIZE = self.TrainParameters.get("BATCH_SIZE", 16)
        num_classes = len(classes)
        for epoch in range(EPOCH):
            totalAccuracy = 0
            for inputs, targets in DATALOADER:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                targets = ChangeMaskOrder(targets, classes)

                #! Mask To One Hot
                targets = targets.long() # Maskeyi uzun tamsayı yap

                # One-hot kodlamalı tensor oluştur
                one_hot_mask = torch.zeros((targets.size(0), num_classes, targets.size(2), targets.size(3)), device=DEVICE)

                # Sınıf indekslerini one-hot kodlamalı tensor haline getir
                one_hot_mask.scatter_(1, targets, 1)

                optimizer.zero_grad()
                
                #TODO inputs = inputs[:, 0:10, :, :]     #! Unet3D
                #TODO inputs = inputs.unsqueeze(1) #! Unet3D
                
                inputs = TRANSFORM_IMAGE(inputs)
                outputs = MODEL(inputs)
                
                #TODO one_hot_mask = one_hot_mask.unsqueeze(1) #! Unet3D

                loss = criterion(outputs, one_hot_mask)
                loss.backward()
                optimizer.step()

                # Accuracy
                class_indices = torch.argmax(outputs, dim=1)  #TODO Unet3D dim2
                class_indices = class_indices.squeeze(1)
                accuracy = 100*((class_indices.flatten() == targets.flatten()).sum() / patch_size**2 /inputs.size(0))
                totalAccuracy += accuracy.item()
                print(f"Epoch {epoch+1}/{EPOCH}, Train Loss: {loss.item()/BATCH_SIZE}, Train Accuracy: {accuracy.item()}")
                try:
                    wandb.log({
                        "epoch": epoch+1,
                        "train_loss": loss.item()/BATCH_SIZE,
                        "train_accuracy": accuracy
                    })
                except Exception as e:
                    print(e)

            # print(f"Epoch {epoch+1}/{EPOCH}, Epoch Accuracy: {totalAccuracy}")


        torch.save(MODEL.state_dict(), "./weight/custom02_unet.pth")
        # wandb.save(".data/wandb/weight/custom02_unet.pth")
        # wandb.finish()
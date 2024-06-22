from Model.Enum import ModelType


TRAIN_DEFAULTS = {
    ModelType.UNET_2D: {
        "EPOCH": 1000,
        "BATCH_SIZE": 16
    },

    ModelType.UNET_3D: {
        "EPOCH": 1000,
        "BATCH_SIZE": 16
    },

    ModelType.RESNET_50: {
        "EPOCH": 1000,
        "BATCH_SIZE": 16
    }

}
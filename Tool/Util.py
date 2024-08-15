import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field, validator
from Tool.DataStorage import GetIndexDatasetPath



class FilePath(BaseModel):
    Path: Annotated[str, Field(..., description="File Path")]
    Type: Annotated[Optional[str], Field(..., description="File Type")] = None

    @validator("Path")                         # v2: field_validator
    def _ValidatePath(cls, value: str):
        _path = value.replace(os.sep, "/")
        _path = os.path.abspath(_path)
        return _path


class DataSourceMeta(BaseModel):
    Scene: Annotated[str, "Scene Name"]
    DataPaths: Annotated[List[FilePath], "Bands FilePath"]
    LabelPaths: Annotated[List[FilePath], "Labels FilePath"]


def ExtractDatasetMeta(data_dir, target_dir) -> List[DataSourceMeta]:
    allData: List[DataSourceMeta] = []
    for root, dirs, files in os.walk(data_dir):
        if root == data_dir:                    # First Depth Level
            scenes = dirs.copy()
        else:
            basename = os.path.basename(root)
            if basename in scenes:              # Second Depth Level
                dataFilePaths = list(map(lambda x: FilePath(Path=os.path.join(root, x)), files))
                targetScenePath = os.path.join(target_dir, basename)
                
                targetFilePaths = [
                    FilePath(Path=_x) for _x in map(lambda x: os.path.join(targetScenePath, x), os.listdir(targetScenePath)) 
                        if os.path.isfile(_x)
                ]               

                allData+= [
                    DataSourceMeta(
                        Scene=basename,
                        DataPaths=dataFilePaths,
                        LabelPaths=targetFilePaths
                    )
                ]

    return allData


def CrateDatasetIndexFile(dataset_source_meta: List[DataSourceMeta], save_dir: str = None, save_file: bool = True) -> dict:
    index = {"data" : {}}
    
    data: DataSourceMeta
    for i, data in enumerate(dataset_source_meta):
        index["data"].update({f"{i}": data.dict()}) #model_dump

    if save_file:
        file_path = os.path.join(save_dir, "index.json")
        with open(file_path, "w") as file:
            json.dump(index, file, indent=4)

    return index


def ReadDatasetFromIndexFile(dataset_dir: str) -> List[DataSourceMeta]:
    file_path = os.path.join(dataset_dir, "index.json")
    with open(file_path, "r") as file:
        jsonData:dict = json.load(file)
        datasetIndexMeta = []
        for key, value in jsonData["data"].items():
            datasetIndexMeta+=[DataSourceMeta(**value)]
        return datasetIndexMeta



if "__main__" == __name__:
    os.environ["DATA_INDEX_FILE"] ="data/dataset/.index"
    DATASET_PATH = GetIndexDatasetPath("LULC_IO_10m")
    DATA_PATH = DATASET_PATH + f"/data/Resample/raster"
    MASK_PATH = DATASET_PATH + f"/mask/raster/"

    dataset = ExtractDatasetMeta(DATA_PATH, MASK_PATH)
    CrateDatasetIndexFile(dataset, save_dir=DATASET_PATH)
    data = ReadDatasetFromIndexFile(DATASET_PATH)

    for x in data:
        print(x.LabelPaths[0])
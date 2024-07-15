import json
from typing import Any, Dict
from pydantic import BaseModel
import os

# os.environ["DATA_INDEX_FILE"] ="C:/Users/DGH/source/repos/Local/data/.index"

class _PyDict(BaseModel):
    paths: Dict[str, Any]

class _IndexData(BaseModel):
    dataset: _PyDict

def _ReadMetaFile(file_path) -> _IndexData:
    with open(file_path, "r") as file:
        jsonData:dict = json.load(file)
        jsonData["path"]=file_path
        return _IndexData(**jsonData)

def _CheckIndexExists():
    INDEX_PATH = os.environ.get("DATA_INDEX_FILE", "./.index")
    if os.path.exists(INDEX_PATH):
        return INDEX_PATH
    return None

def ListIndexDatasets() -> _PyDict:
    INDEX_PATH = _CheckIndexExists()
    return _ReadMetaFile(INDEX_PATH).dataset

def GetIndexDatasetPath(dataset_name):
    datasets: _PyDict = ListIndexDatasets()
    return datasets.paths.get(dataset_name, "")


if "__main__" == __name__:
    datasetList = ListIndexDatasets()
    path = GetIndexDatasetPath("MiningArea01")
    print(path)
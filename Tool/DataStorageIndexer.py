import json
import os
from typing import Any, Dict, List

from pydantic import BaseModel

os.environ["PYTHON_SCRIPT_DATASET_PATH"] = "/home/blitzkrieg/source/repos/Workshop/Project/Github/Python/LandUseLandCoverClassification/data/dataset"

PYTHON_SCRIPT_DATASET_PATH = os.getenv("PYTHON_SCRIPT_DATASET_PATH") or "./"
INDEX_OUT_PATH = os.path.join(PYTHON_SCRIPT_DATASET_PATH, ".index")

class _MetaData(BaseModel):
    id: str = None
    hash: str = None
    date: str = None
    name: str
    description: str = None
    tags: str = None
    path: str

class _PyDict(BaseModel):
    paths: Dict[str, Any]

class _IndexData(BaseModel):
    dataset: _PyDict

def __ReadMetaFile(file_path):
    with open(file_path, "r") as file:
        jsonData:dict = json.load(file)
        purePath = os.path.dirname(file_path).replace(os.sep, '/')
        jsonData["path"]=purePath
        return _MetaData(**jsonData)

def __FindMetaFiles(directory, searchString=".meta"):
    metaFiles = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(searchString):
                metaFiles+=[os.path.join(os.path.abspath(root) , file)]
    return metaFiles

def __CreateIndexFileFromMetaFiles(meta_files: List[_MetaData], out_path=INDEX_OUT_PATH):
    existPaths = {f"{meta.name}" : meta.path for meta in meta_files}

    with open(out_path, "w") as file:
        datasetPaths = _PyDict(paths=existPaths)
        indexData = _IndexData(dataset=datasetPaths)
        try:
            dump = indexData.model_dump()
        except Exception as err:
            dump = indexData.dict()
        
        json.dump(dump, file, indent=4)

    return dump


def UpdateIndexData(directory=PYTHON_SCRIPT_DATASET_PATH):
    metas = [__ReadMetaFile(filePath) for filePath in __FindMetaFiles(directory)]
    dump = __CreateIndexFileFromMetaFiles(metas, INDEX_OUT_PATH)
    return dump


if "__main__" == __name__:
    UpdateIndexData(directory=PYTHON_SCRIPT_DATASET_PATH)

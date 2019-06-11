import pathlib,os

def createFolderIfNotExists(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

def folderExists(folder):
    return os.path.isdir(folder)

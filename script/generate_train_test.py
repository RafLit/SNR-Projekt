import os
import pathlib
import random
import math
import shutil
if __name__ == '__main__':
    trainTestSplit = 0.7
    acceptPath = os.path.join('..','data','accept')
    rejectPath = os.path.join('..','data','reject')
    acceptTrainPath = os.path.join('..','data','train','accept')
    acceptTestPath = os.path.join('..','data','test','accept')
    rejectTrainPath = os.path.join('..','data','train','reject')
    rejectTestPath = os.path.join('..','data','test','reject')
    outDirs = [acceptTrainPath, acceptTestPath, rejectTrainPath, rejectTestPath]
    if not os.path.exists(acceptPath) or not os.path.exists('../data/reject'):
        raise FileNotFoundError()
    acceptFiles = [os.path.join(acceptPath, file) for file in os.listdir(acceptPath)]
    rejectFiles = [os.path.join(rejectPath, file) for file in os.listdir(rejectPath)]
    random.shuffle(acceptFiles)
    random.shuffle(rejectFiles)
    acceptSplit = math.ceil(len(acceptFiles)*trainTestSplit)
    rejectSplit = math.ceil(len(rejectFiles)*trainTestSplit)
    acceptTrainFiles = acceptFiles[:acceptSplit]
    acceptTestFiles = acceptFiles[acceptSplit:]
    rejectTrainFiles = rejectFiles[:rejectSplit]
    rejectTestFiles = rejectFiles[rejectSplit:]
    outFiles = [acceptTrainFiles, acceptTestFiles, rejectTrainFiles, rejectTestFiles]

    for directory in outDirs:
        if not os.path.isdir(directory):
            os.makedirs(directory)
        for file in os.listdir(directory):
            if os.path.isfile(file):
                os.remove(file)
    for path, files in zip(outDirs, outFiles):
        for file in files:
            if not os.path.isfile(file):
                continue
            outPath = os.path.join(path, os.path.split(file)[1])
            print(file)
            print(outPath)
            shutil.copy(file, outPath)





#!/usr/bin/env python3
import os
import random
import math
import shutil

trainTestSplit = 0.7

if __name__ == '__main__':
    acceptPath = os.path.join('..','data','accept')
    rejectPath = os.path.join('..','data','reject')
    acceptTrainPath = os.path.join('..','data','train','accept')
    acceptTestPath = os.path.join('..','data','test','accept')
    rejectTrainPath = os.path.join('..','data','train','reject')
    rejectTestPath = os.path.join('..','data','test','reject')
    outDirs = [acceptTrainPath, acceptTestPath, rejectTrainPath, rejectTestPath]
    if not os.path.exists(acceptPath) or not os.path.exists(rejectPath):
        raise FileNotFoundError('couldnt find any data!')
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
            shutil.copy(file, outPath)





# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 19:08:53 2014

@author: Sławomir Figiel
"""
from os.path import isfile, join, isdir, splitext
from os import listdir

def pathReduction(paths):
    files = []
    for path in paths:
        if isdir(path):
            files.extend(filesInDir(path))
        else:
            files.append(path) 
    return files

def sortPaths(pathsFiles, *extensions):
    '''

    '''
    dictExtFiles = dict((ext, []) for ext in extensions)
    dictKeys = dictExtFiles.keys()

    for f in pathsFiles:
        ext = splitext(f)[1]
        if ext in dictKeys:
            dictExtFiles[ext].append(f)       
    return dictExtFiles
    
def filesInDir(dir_path):
    full = [join(dir_path, f) for f in listdir(dir_path)]
    return [f for f in full if isfile(f)]

if __name__ == '__main__':
    pass
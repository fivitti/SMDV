# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 19:08:53 2014
@author: Slawomir Figiel

Module with the methods working on file paths/
"""
from os.path import isfile, join, isdir, splitext
from os import listdir

def path_reduction(paths):
    '''
    Method process paths. If in paths is directory, method searches it 
    for the files. Returns list of files path with paths and files in 
    directories.
    
    Parameters
    ==========
    paths : enumerate
        list of files
    
    Returns
    =======
    paths : list
        list of path files in paths and files in directory in paths
        
    Examples
    ========
    >>> paths = [data/file1.txt, data/file2.pdf, dir/]
    >>> path_reduction(paths)
        [data/file1.txt, data/file2.pdf, dir/file3.map, dir/file4.rtf]
    '''
    files = []
    for path in paths:
        if isdir(path):
            files.extend(files_in_dir(path))
        else:
            files.append(path) 
    return files

def sort_paths(paths_files, *extensions):
    '''
    Method sorts paths after file extension and returns a dictionary whose 
    keys are the extension and the values list of path.
    
    Parameters
    ==========
    paths_files : list paths files
        paths to sort
    *extensions : strings
        extensions that will be included in the dictionary
    
    Returns
    =======
    sort paths : dict
        dictionary whose  keys are the extension and values list of
        path
    
    Examples
    ========
    >>> paths = ['dir/file1.txt', 'file2.pdf', 'dir/file3.txt',
                 'file4.rtf', 'file5']
    >>> sort_paths(paths, 'txt', 'pdf')
        {'txt' : ['dir/file1.txt', 'dir/file3.txt'], 'pdf' : [file2.pdf]}
    '''
    dictExtFiles = dict((ext, []) for ext in extensions)
    dictKeys = dictExtFiles.keys()

    for f in paths_files:
        ext = splitext(f)[1]
        if ext in dictKeys:
            dictExtFiles[ext].append(f)       
    return dictExtFiles
    
def files_in_dir(dir_path):
    '''
    Method return full path files in directory.
    
    Parameters
    ==========
    dir_path : string
        directory
        
    Returns
    =======
    files paths : list
        List of full files path in directory.
        
    Examples
    ========
    >>> files_in_dir(dir)
        ['dir/file1.txt', 'dir/file2', 'dir/file3.pdf']
    '''
    full = [join(dir_path, f) for f in listdir(dir_path)]
    return [f for f in full if isfile(f)]

if __name__ == '__main__':
    pass
    
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:02:08 2014

@author: Sławomir Figiel
"""
from pycuda.compiler import SourceModule
from os.path import join as path_join

KERNELS_PATH = "kernels"

def convertString(string, **kwargs):
    s = string
    for name, value in kwargs.items():
        value = str(value)
        s = s.replace("{{"+name+"}}", value)
        s = s.replace("{{ "+name+"}}", value)
        s = s.replace("{{"+name+" }}", value)
        s = s.replace("{{ "+name+" }}", value)
    return s
        
def get_cuda_ellpack():
    kernel_info = {'file_' : 'ellpack_kernel.c',
                   'kernel' : 'EllpackFormatKernel',
                   'texref' : 'mainVecTexRef'}
    with open(path_join(KERNELS_PATH, kernel_info['file_'])) as file_:
        tpl = file_.read()
    mod = SourceModule(tpl)
    kernel = mod.get_function(kernel_info['kernel'])
    texref = mod.get_texref(kernel_info['texref'])
    return (kernel, texref)
    
def get_cuda_sliced(sh_cache_size, threadPerRow = 2):
    kernel_info = {'file_' : 'sliced_kernel.c',
                   'kernel' : 'SlicedEllpackFormatKernel',
                   'texref' : 'mainVecTexRef'}
    with open(path_join(KERNELS_PATH, kernel_info['file_'])) as file_:
        tpl = file_.read()
    tpl = convertString(tpl, sh_cache_size=sh_cache_size,
                        threadPerRow=threadPerRow)
    mod = SourceModule(tpl)
    kernel = mod.get_function(kernel_info['kernel'])
    texref = mod.get_texref(kernel_info['texref'])
    return (kernel, texref)

def get_cuda_sertilp(shDot_size = None, threadPerRow = 2, sliceSize = 32, prefetch = 2):
    kernel_info = {'file_' : 'sertilp_kernel.c',
                   'kernel' : 'rbfSERTILP_old',
                   'texref' : 'mainVecTexRef'}
    with open(path_join(KERNELS_PATH, kernel_info['file_'])) as file_:
        tpl = file_.read()
    if shDot_size is None:
        shDot_size = threadPerRow * sliceSize
    tpl = convertString(tpl, shDot_size = shDot_size, threadPerRow = threadPerRow, sliceSize = sliceSize, prefetch = prefetch) 
    mod = SourceModule(tpl)
    kernel = mod.get_function(kernel_info['kernel'])
    texref = mod.get_texref(kernel_info['texref'])
    return (kernel, texref)

def get_cuda_ertilp(block_sice, threadPerRow, prefetch):
    kernel_info = {'file_' : 'ertilp_kernel.c',
                   'kernel' : 'rbfERTILP',
                   'texref' : 'labelsTexRef'}
    with open(path_join(KERNELS_PATH, kernel_info['file_'])) as file_:
        tpl = file_.read()
    prefetch_init_tab = '{' + \
                        ', '.join('0' for i in range(prefetch)) + \
                        '}'
    tpl = convertString(tpl, BLOCK_SIZE = block_sice, THREADS_ROW = threadPerRow, PREFETCH_SIZE = prefetch, PREFETCH_INIT_TAB = prefetch_init_tab)
    mod = SourceModule(tpl)
    kernel = mod.get_function(kernel_info['kernel'])
    texref = mod.get_texref(kernel_info['texref'])
    return (kernel, texref)
    
def getCsrCudaCode(block_size = 128, warp_size = 32):
    kernel_info = {'file_' : 'csr_kernel.c'}
    with open(path_join(KERNELS_PATH, kernel_info['file_'])) as file_:
        tpl = file_.read()
    tpl = convertString(tpl, BLOCK_SIZE = block_size, WARP_SIZE = warp_size)
    return tpl
if __name__ == "__main__":
    p = getSertilpCudaCode()
    print p
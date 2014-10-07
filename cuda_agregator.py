# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:02:08 2014
@author: Sławomir Figiel

Module provides the methods returns metaprogrammed and compiled CUDA code
for matrix multiplication on GPU to formats:
    * CSR (Scipy)
    * Ellpack
    * Sliced Ellpack
    * SERTILP
    * ERTILP
Make sure that the constant "KERNELS_PATH" indicates
a folder with files kernels.
"""
from pycuda.compiler import SourceModule
from os.path import join as path_join

KERNELS_PATH = "kernels"

def convert_string(string, **kwargs):
    '''
    Method replace keys of kwargs with their values in string.
    Modeled on the template from Jinja2.

    Parameters
    ==========
    string : string
        String to change
    kwargs : dict
        Keys the dictionary will be replaced with their values.
        Keys in string must be in format {{ key }}.

    Returns
    =======
    Changed string in which dictionary keys replaced with values.

    Examples
    ========
    >>> s = 'Lorem {{ f1 }} dolor sit {{ f2 }}, consectetur adipiscing elit'
    >>> convert_string(s, f1='ipsum', f2='amet')
    Lorem ipsum dolor sit amet, consectetur adipiscing elit
    '''
    str_ = string
    for name, value in kwargs.items():
        value = str(value)
        str_ = str_.replace("{{"+name+"}}", value)
        str_ = str_.replace("{{ "+name+"}}", value)
        str_ = str_.replace("{{"+name+" }}", value)
        str_ = str_.replace("{{ "+name+" }}", value)
    return str_

def get_cuda_ellpack():
    '''
    Method read ELLPACK CUDA code from file, compile it, and returns
    ready kernel and texture.

    Returns
    =======
    Tuple of compiled kernel and texture
    '''
    kernel_info = {'file_' : 'ellpack_kernel.c',
                   'kernel' : 'EllpackFormatKernel',
                   'texref' : 'mainVecTexRef'}
    with open(path_join(KERNELS_PATH, kernel_info['file_'])) as file_:
        tpl = file_.read()
    mod = SourceModule(tpl)
    kernel = mod.get_function(kernel_info['kernel'])
    texref = mod.get_texref(kernel_info['texref'])
    return (kernel, texref)

def get_cuda_sliced(sh_cache_size, threads_per_row=2):
    '''
    Method read SLICED CUDA code from file, build it with arguments,
    compile it, and returns ready kernel and texture.
    The parameters of method must be equal parameters of converted matrix,
    which is multiplied.

    Parameters
    ==========
    sh_cache_size : int
        Size of cache array. For Sliced format must be equal
        threads per row * slice size. If get another value execute badly.
    threads_per_row : int > 0 (Recommended 2, 4 or 8)
        Threads per row

    Returns
    =======
    Tuple of compiled kernel and texture
    '''
    kernel_info = {'file_' : 'sliced_kernel.c',
                   'kernel' : 'SlicedEllpackFormatKernel',
                   'texref' : 'mainVecTexRef'}
    with open(path_join(KERNELS_PATH, kernel_info['file_'])) as file_:
        tpl = file_.read()

    tpl = convert_string(tpl, sh_cache_size=sh_cache_size,
                         threadPerRow=threads_per_row)

    mod = SourceModule(tpl)
    kernel = mod.get_function(kernel_info['kernel'])
    texref = mod.get_texref(kernel_info['texref'])
    return (kernel, texref)

def get_cuda_sertilp(sh_dot_size=None, threads_per_row=2,
                     slice_size=32, prefetch=2):
    '''
    Method read SERTILP CUDA code from file, build it with arguments,
    compile it, and returns ready kernel and texture.
    The parameters of method must be equal parameters of converted matrix,
    which is multiplied.

    Parameters
    ==========
    sh_dot_size : int
        Size of cache array. For Sertilp format must be equal
        threads per row * slice size. If get another value execute badly.
        If get None calculates this automatically.
    threads_per_row : int > 0 (Recommended 2, 4 or 8)
        Threads per row
    slice_size : int (Recommended multiple 2)
        Slice simple size.
    prefetch : int (recommended 2, 4 or 8)
        Number of requests for access to data notified in advance.

    Returns
    =======
    Tuple of compiled kernel and texture
    '''
    kernel_info = {'file_' : 'sertilp_kernel.c',
                   'kernel' : 'rbfSERTILP_old',
                   'texref' : 'mainVecTexRef'}
    with open(path_join(KERNELS_PATH, kernel_info['file_'])) as file_:
        tpl = file_.read()

    if sh_dot_size is None:
        sh_dot_size = threads_per_row * slice_size
    tpl = convert_string(tpl, shDot_size=sh_dot_size,
                         threadPerRow=threads_per_row,
                         sliceSize=slice_size, prefetch=prefetch)

    mod = SourceModule(tpl)
    kernel = mod.get_function(kernel_info['kernel'])
    texref = mod.get_texref(kernel_info['texref'])
    return (kernel, texref)

def get_cuda_ertilp(block_size, threads_per_row, prefetch):
    '''
    Method read ERTILP CUDA code from file, build it with arguments,
    compile it, and returns ready kernel and texture.
    The parameters of method must be equal parameters of converted matrix,
    which is multiplied.

    Parameters
    ==========
    block_size : int (Recommended 128 or 256)
        Size of block
    threads_per_row : int > 0 (Recommended 2, 4 or 8)
        Threads per row
    prefetch : int (recommended 2, 4 or 8)
        Number of requests for access to data notified in advance.

    Returns
    =======
    Tuple of compiled kernel and texture
    '''
    kernel_info = {'file_' : 'ertilp_kernel.c',
                   'kernel' : 'rbfERTILP',
                   'texref' : 'labelsTexRef'}
    with open(path_join(KERNELS_PATH, kernel_info['file_'])) as file_:
        tpl = file_.read()

    prefetch_init_tab = '{' + \
                        ', '.join('0' for i in range(prefetch)) + \
                        '}'
    tpl = convert_string(tpl, BLOCK_SIZE=block_size,
                         THREADS_ROW=threads_per_row,
                         PREFETCH_SIZE=prefetch,
                         PREFETCH_INIT_TAB=prefetch_init_tab)

    mod = SourceModule(tpl)
    kernel = mod.get_function(kernel_info['kernel'])
    texref = mod.get_texref(kernel_info['texref'])
    return (kernel, texref)

def get_cuda_csr(block_size=128, warp_size=32):
    '''
    Method read ERTILP CUDA code from file, build it with arguments,
    compile it, and returns ready kernel and texture.
    The parameters of method must be equal parameters of converted matrix,
    which is multiplied.

    Parameters
    ==========
    block_size : int (Recommended 128 or 256)
        Size of block
    warp_size : int > 0 (Recommended 32)
        Size of warp. This value depends on the specifications
        of the graphics card.
    Returns
    =======
    Tuple of compiled kernel and texture
    '''
    kernel_info = {'file_' : 'csr_kernel.c',
                   'kernel' : 'rbfCsrFormatKernel',
                   'texref' : 'mainVecTexRef'}
    with open(path_join(KERNELS_PATH, kernel_info['file_'])) as file_:
        tpl = file_.read()

    tpl = convert_string(tpl, BLOCK_SIZE=block_size, WARP_SIZE=warp_size)

    mod = SourceModule(tpl)
    kernel = mod.get_function(kernel_info['kernel'])
    texref = mod.get_texref(kernel_info['texref'])
    return (kernel, texref)

if __name__ == "__main__":
    pass

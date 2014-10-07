# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:35:44 2014

@author: Sławomir Figiel
Module provides the methods for multiplication on GPU matrix to formats:
    * CSR (Scipy)
    * Ellpack
    * Sliced Ellpack
    * SERTILP
    * ERTILP
 
There is also method for multiplication using Numpy.dot function
for comparison.
"""
from matrixformat import convert_to_ellpack, convert_to_sliced, \
                         convert_to_sertilp, convert_to_ertilp, \
                         convert_to_scipy_csr
import cudaAgregator

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy
from math import ceil

start = cuda.Event()
end = cuda.Event()

def multiply_cpu(matrix, vector, repeat=1):
    '''
    Method multiply matrix by vector using Numpy module.
    Calculation executed only on the processor.

    Parameters
    ==========
    matrix : Scipy matrix or numpy array
        Matrix to multiplication.
    vector : numpy array
        Vector to multiplication. His length must equal number of columns
        matrix.
    repeat : int > 0
        Number of repetitions multiplications. It has no effect on
        result. Specifies the length of returned list of execution times.

    Returns
    =======
    Tuple of result multiplication and list of execution times.
    '''
    if len(vector) != matrix.shape[1]:
        raise ArithmeticError('Length of the vector is not equal to the'
                              'number of columns of the matrix.')
    time_list = []

    for _ in range(repeat):
        start.record()
        result = matrix.dot(vector)
        end.record()
        end.synchronize()
        time_list.append(start.time_till(end))

    return (result, time_list)

def multiply_csr(matrix, vector, block_size, repeat=1):
    '''
    Method multiply matrix by vector using CUDA module for CSR.
    Calculation executed on nVidia GPU.

    Parameters
    ==========
    matrix : Scipy matrix or numpy array
        Matrix to multiplication.
    vector : numpy array
        Vector to multiplication. His length must equal number of columns
        matrix.
    block_size : int (recommended 128 or 256)
        Size of block CUDA.
    repeat : int > 0
        Number of repetitions multiplications. It has no effect on
        result. Specifies the length of returned list of execution times.

    Returns
    =======
    Tuple of result multiplication and list of execution times.
    '''
    if len(vector) != matrix.shape[1]:
        raise ArithmeticError('Length of the vector is not equal to the'
                              'number of columns of the matrix.')
    matrix = convert_to_scipy_csr(matrix)
    data = numpy.array(matrix.data, dtype=numpy.float32)
    indices = numpy.array(matrix.indices, dtype=numpy.int32)
    indptr = numpy.array(matrix.indptr, dtype=numpy.int32)
    data = cuda.to_device(data)
    indices = cuda.to_device(indices)
    indptr = cuda.to_device(indptr)
    num_rows = matrix.shape[0]
    result = numpy.zeros(num_rows, dtype=numpy.float32)
    time_list = []

    grid_size = int(numpy.ceil((num_rows+0.0)/block_size))
    block = (block_size, 1, 1)
    grid = (grid_size, 1)
    g_vector = cuda.to_device(vector)
    num_rows = numpy.int32(num_rows)

    mod = SourceModule(cudaAgregator.getCsrCudaCode(block_size=block_size))
    kernel = mod.get_function("rbfCsrFormatKernel")
    texref = mod.get_texref("mainVecTexRef")
    texref.set_address(g_vector, vector.nbytes)
    tex = [texref]

    for _ in range(repeat):
        start.record()
        kernel(data,
               indices,
               indptr,
               cuda.Out(result),
               num_rows,
               block=block,
               grid=grid,
               texrefs=tex)
        end.record()
        end.synchronize()
        time_list.append(start.time_till(end))
    return (result, time_list)

def multiply_ellpack(matrix, vector, block_size=128, repeat=1):
    '''
    Method multiply matrix by vector using CUDA module for ELLPACK Format.
    Calculation executed on nVidia GPU.

    Parameters
    ==========
    matrix : Scipy matrix or numpy array
        Matrix to multiplication.
    vector : numpy array
        Vector to multiplication. His length must equal number of columns
        matrix.
    block_size : int (recommended 128 or 256)
        Size of block CUDA.
    repeat : int > 0
        Number of repetitions multiplications. It has no effect on
        result. Specifies the length of returned list of execution times.

    Returns
    =======
    Tuple of result multiplication and list of execution times.
    '''
    num_rows, num_cols = matrix.shape
    if len(vector) != num_cols:
        raise ArithmeticError('Length of the vector is not equal to the '
                              'number of columns of the matrix.')
    matrix = convert_to_ellpack(matrix)
    vals = cuda.to_device(matrix[0])
    col_idx = cuda.to_device(matrix[1])
    rows_length = cuda.to_device(matrix[2])
    result = numpy.zeros(num_rows, dtype=numpy.float32)
    num_rows = numpy.int32(num_rows)
    time_list = []

    grid_size = int(numpy.ceil((num_rows+0.0)/block_size))
    block = (block_size, 1, 1)
    grid = (grid_size, 1)
    g_vector = cuda.to_device(vector)

    kernel, texref = cudaAgregator.get_cuda_ellpack()
    texref.set_address(g_vector, vector.nbytes)
    tex = [texref]

    for _ in range(repeat):
        start.record()
        kernel(vals,
               col_idx,
               rows_length,
               cuda.Out(result),
               num_rows,
               block=block,
               grid=grid,
               texrefs=tex)
        end.record()
        end.synchronize()
        time_list.append(start.time_till(end))
    return (result, time_list)

def multiply_sliced(matrix, vector, align,
                    slice_size, threads_per_row, repeat=1):
    '''
    Method multiply matrix by vector using CUDA module for SLICED ELLPACK
    Format.
    Calculation executed on nVidia GPU.

    Parameters
    ==========
    matrix : Scipy matrix or numpy array
        Matrix to multiplication.
    vector : numpy array
        Vector to multiplication. His length must equal number of columns
        matrix.
    align : int > 0 (recommended multiple 2)
        Constant to calculation and set align of the matrix. Will be used
        a formula to normalize:
            ceil( (slice_size * threads_per_row) / align) * align
    slice_size : int > 0 (recommended multiple 2)
        Size of simple slice
    threads_per_row : int > 0 (recommended 2, 4 or 8)
        Threads per row
    repeat : int > 0
        Number of repetitions multiplications. It has no effect on
        result. Specifies the length of returned list of execution times.

    Returns
    =======
    Tuple of result multiplication and list of execution times.
    '''
    num_rows, num_cols = matrix.shape
    if len(vector) != num_cols:
        raise ArithmeticError('Length of the vector is not equal to'
                              'the number of columns of the matrix.')
    align = int(ceil((slice_size*threads_per_row*1.0)/align)*align)
    matrix = convert_to_sliced(matrix, threads_per_row=threads_per_row,
                               slice_size=slice_size, align=align)
    vals = cuda.to_device(matrix[0])
    col_idx = cuda.to_device(matrix[1])
    rows_length = cuda.to_device(matrix[2])
    slices_start = cuda.to_device(matrix[3])
    result = numpy.zeros(num_rows, dtype=numpy.float32)
    num_rows = numpy.int32(num_rows)
    align = numpy.int32(align)
    slice_size = numpy.int32(slice_size)
    time_list = []

    block_size = int(threads_per_row * slice_size)
    grid_size = int(numpy.ceil((num_rows*threads_per_row+0.0)/block_size))
    block = (block_size, 1, 1)
    grid = (grid_size, 1)
    g_vector = cuda.to_device(vector)

    kernel, texref = cudaAgregator.get_cuda_sliced(
                        sh_cache_size=threads_per_row*slice_size,
                        threadPerRow=threads_per_row)
    texref.set_address(g_vector, vector.nbytes)
    tex = [texref]

    for _ in range(repeat):
        start.record()
        kernel(vals,
               col_idx,
               rows_length,
               slices_start,
               cuda.Out(result),
               num_rows,
               align,
               block=block,
               grid=grid,
               texrefs=tex)
        end.record()
        end.synchronize()
        time_list.append(start.time_till(end))
    return (result, time_list)

def multiply_sertilp(matrix, vector, align, slice_size,
                     threads_per_row, prefetch=2, repeat=1):
    '''
    Method multiply matrix by vector using CUDA module for SERTILP
    Format.
    Calculation executed on nVidia GPU.

    Parameters
    ==========
    matrix : Scipy matrix or numpy array
        Matrix to multiplication.
    vector : numpy array
        Vector to multiplication. His length must equal number of columns
        matrix.
    align : int > 0 (recommended multiple 2)
        Constant to calculation and set align of the matrix. Will be used
        a formula to normalize:
            ceil( (slice_size * threads_per_row) / align) * align
    slice_size : int > 0 (recommended multiple 2)
        Size of simple slice
    threads_per_row : int > 0(recommended 2, 4 or 8)
        Threads per row
    prefetch : int > 0 (reommended 2, 4 or 8)
        Number of requests for access to data notified in advance.
    repeat : int > 0
        Number of repetitions multiplications. It has no effect on
        result. Specifies the length of returned list of execution times.

    Returns
    =======
    Tuple of result multiplication and list of execution times.
    '''
    num_rows, num_cols = matrix.shape
    if len(vector) != num_cols:
        raise ArithmeticError('Length of the vector is not equal to'
                              'the number of columns of the matrix.')
    align = int(ceil((slice_size*threads_per_row*1.0)/align)*align)
    matrix = convert_to_sertilp(matrix, threads_per_row=threads_per_row,
                                slice_size=slice_size, prefetch=prefetch,
                                align=align)
    # Array matrix[2] is converted explicitly to an array of float,
    # because it is element of a tuple. Without this
    # function "ceil" does not work properly.
    rows_length = numpy.array( \
                    numpy.ceil(
                        numpy.float32(matrix[2]) \
                        / (threads_per_row*prefetch)),
                    dtype=numpy.int32)
    vals = cuda.to_device(matrix[0])
    col_idx = cuda.to_device(matrix[1])
    rows_length = cuda.to_device(rows_length)
    slices_start = cuda.to_device(matrix[3])
    result = numpy.zeros(num_rows, dtype=numpy.float32)
    num_rows = numpy.int32(num_rows)
    align = numpy.int32(align)
    time_list = []

    block_size = threads_per_row * slice_size
    grid_size = int(numpy.ceil((num_rows*threads_per_row+0.0)/block_size))
    block = (block_size, 1, 1)
    grid = (grid_size, 1)
    g_vector = cuda.to_device(vector)

#    mod = SourceModule( \
#            cudaAgregator.getSertilpCudaCode(
#                threadPerRow=threads_per_row,
#                sliceSize=slice_size,
#                prefetch=prefetch))
#    kernel = mod.get_function("rbfSERTILP_old")
#    texref = mod.get_texref("mainVecTexRef")
    kernel, texref = cudaAgregator.get_cuda_sertilp(
                        threadPerRow=threads_per_row,
                        sliceSize=slice_size,
                        prefetch=prefetch)
    texref.set_address(g_vector, vector.nbytes)
    tex = [texref]

    for _ in range(repeat):
        start.record()
        kernel(vals,
               col_idx,
               rows_length,
               slices_start,
               cuda.Out(result),
               num_rows,
               align,
               block=block,
               grid=grid,
               texrefs=tex)
        end.record()
        end.synchronize()
        time_list.append(start.time_till(end))
    return (result, time_list)

def multiply_ertilp(matrix, vector, threads_per_row=2,
                    prefetch=2, block_size=128, repeat=1):
    '''
    Method multiply matrix by vector using CUDA module for ERTILP
    Format.
    Calculation executed on nVidia GPU.

    Parameters
    ==========
    matrix : Scipy matrix or numpy array
        Matrix to multiplication.
    vector : numpy array
        Vector to multiplication. His length must equal number of columns
        matrix.
    threads_per_row : int > 0(recommended 2, 4 or 8)
        Threads per row
    prefetch : int > 0 (reommended 2, 4 or 8)
        Number of requests for access to data notified in advance.
    block_size : int (recommended 128 or 256)
        Size of block CUDA.
    repeat : int > 0
        Number of repetitions multiplications. It has no effect on
        result. Specifies the length of returned list of execution times.

    Returns
    =======
    Tuple of result multiplication and list of execution times.
    '''
    num_rows, num_cols = matrix.shape
    if len(vector) != num_cols:
        raise ArithmeticError('Length of the vector is not equal to '
                              'the number of columns of the matrix.')
    matrix = convert_to_ertilp(matrix, prefetch=prefetch,
                               threads_per_row=threads_per_row)
    # Array matrix[2] is converted explicitly to an array of float,
    # because it is element of a tuple. Without this
    # function "ceil" does not work properly.
    rows_length = numpy.array( \
                    numpy.ceil(
                        numpy.float32(matrix[2]) \
                        / (threads_per_row*prefetch)),
                    dtype=numpy.int32)
    rows_length = cuda.to_device(rows_length)
    vals = cuda.to_device(matrix[0])
    col_idx = cuda.to_device(matrix[1])
    result = numpy.zeros(num_rows, dtype=numpy.float32)
    num_rows = numpy.int32(num_rows)
    time_list = []

    grid_size = int(numpy.ceil((num_rows*threads_per_row+0.0)/block_size))
    block = (block_size, 1, 1)
    grid = (grid_size, 1)
    g_vector = cuda.to_device(vector)

    mod = SourceModule(cudaAgregator.getErtilpCudaCode( \
                       block_sice=block_size,
                       threadPerRow=threads_per_row,
                       prefetch=prefetch))
    kernel = mod.get_function("rbfERTILP")
    texref = mod.get_texref("labelsTexRef")
    texref.set_address(g_vector, vector.nbytes)
    tex = [texref]

    for _ in range(repeat):
        start.record()
        kernel(vals,
               col_idx,
               rows_length,
               cuda.Out(result),
               num_rows,
               block=block,
               grid=grid,
               texrefs=tex)
        end.record()
        end.synchronize()
        time_list.append(start.time_till(end))

    return (result, time_list)
    
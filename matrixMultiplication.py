# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:35:44 2014

@author: Sławomir Figiel
"""
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import numpy
from math import ceil
 
from matrixFormat import convert_to_ellpack, convert_to_sliced, \
                         transform_to_sertilp, \
                         transform_to_ertilp, \
                         convert_to_scipy_csr
import cudaAgregator

start = cuda.Event()
end = cuda.Event()

def multiplyCPU(matrix, vector, repeat = 1):
#    wektor = numpy.arange(1, matrix.shape[1]+1, dtype=numpy.float32)
    if len(vector) != matrix.shape[1]:
        raise ArithmeticError('Length of the vector is not equal to the number of columns of the matrix.')
    timeList = []
    
    for i in range(repeat):
        start.record()
        wynik = matrix.dot(vector)
        end.record()
        end.synchronize()
        timeList.append(start.time_till(end))
    
    return (wynik, timeList)
    
def multiplyCsr(matrix, vector, block_size, repeat=1):
        if len(vector) != matrix.shape[1]:
            raise ArithmeticError('Length of the vector is not equal to the number of columns of the matrix.')
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
        
        gridSize = int(numpy.ceil((num_rows+0.0)/block_size))  
        block=(block_size,1,1)
        grid=(gridSize,1)                    
        g_wektor = cuda.to_device(vector)
        num_rows = numpy.int32(num_rows)
        
        mod = SourceModule(cudaAgregator.getCsrCudaCode(block_size=block_size))
        kernel = mod.get_function("rbfCsrFormatKernel")
        texref = mod.get_texref("mainVecTexRef")
        texref.set_address(g_wektor, vector.nbytes)
        tex = [texref]
        
        for i in range(repeat):
            start.record()
            kernel(data, \
                    indices, \
                    indptr, \
                    cuda.Out(result), \
                    num_rows, \
                    block=block, \
                    grid=grid, \
                    texrefs=tex)
            end.record()
            end.synchronize()
            time_list.append(start.time_till(end))
        
        return (result, time_list)
        
def multiplyELL(macierz, vector, repeat = 1, blockSize = 128): 
    if len(vector) != macierz.shape[1]:
        raise ArithmeticError('Length of the vector is not equal to the number of columns of the matrix.')
    mac = convert_to_ellpack(macierz)
    vals = cuda.to_device(mac[0])
    colIdx = cuda.to_device(mac[1])
    rowLength = cuda.to_device(mac[2])
    
    wierszeMacierzy, kolumnyMacierzy = macierz.shape
    wektor = vector
    wynik = numpy.zeros(wierszeMacierzy, dtype=numpy.float32)
    numRows = numpy.int32(wierszeMacierzy)
    
    ### Przygotowanie stałych czasu ###
    timeList = []
    ###
    
    ### Przygotowanie stałych CUDA ###
    gridSize = int(numpy.ceil((wierszeMacierzy+0.0)/blockSize))  
    block=(blockSize,1,1)
    grid=(gridSize,1)                    
    g_wektor = cuda.to_device(wektor)
    ###
    
    ### Przygotowanie funkcji i tekstury dla EllPack ###
    modELL = SourceModule(cudaAgregator.getELLCudaCode()) 
    kernelELL = modELL.get_function("EllpackFormatKernel")
    texrefELL = modELL.get_texref("mainVecTexRef")    
    
    texrefELL.set_address(g_wektor, wektor.nbytes)
    texELL = [texrefELL]
    ###
    
    
    for i in range(repeat):
        start.record()
        kernelELL(vals, \
                colIdx, \
                rowLength, \
                cuda.Out(wynik), \
                numRows, \
                block=block, \
                grid=grid, \
                texrefs=texELL)
        end.record()
        end.synchronize()
        timeList.append(start.time_till(end))
    
    return (wynik, timeList)
    
def multiplySlicedELL(macierz, vector, alignConst, sliceSize, threadPerRow, repeat = 1):   
    if len(vector) != macierz.shape[1]:
        raise ArithmeticError('Length of the vector is not equal to the number of columns of the matrix.')
    ### Przygotowanie macierzy SlicedEllPack ###
    align = int(ceil((sliceSize*threadPerRow*1.0)/alignConst)*alignConst)
    mac = convert_to_sliced(macierz, threads_per_row=threadPerRow, slice_size=sliceSize, align=align)
    vals = cuda.to_device(mac[0])
    colIdx = cuda.to_device(mac[1])
    rowLength = cuda.to_device(mac[2])
    sliceStart = cuda.to_device(mac[3])
    
    wierszeMacierzy, kolumnyMacierzy = macierz.shape
    wektor = vector  
    wynik = numpy.zeros(wierszeMacierzy, dtype=numpy.float32)
    numRows = numpy.int32(wierszeMacierzy)
    ###
    
    ### Przygotowanie stalych wlasciwych dla SlicedELL
    align = numpy.int32(align)
    sliceSize = numpy.int32(sliceSize)
    ###
    
    ### Przygotowanie stałych czasu ###
    timeList = []
    ###    
    
    ### Przygotowanie stałych CUDA ###
    blockSize = int(threadPerRow * sliceSize);
    gridSize = int(numpy.ceil((wierszeMacierzy*threadPerRow+0.0)/blockSize)) 
    block=(blockSize,1,1)
    grid=(gridSize,1)                    
    g_wektor = cuda.to_device(wektor)
    ###
    
    ### Przygotowanie funkcji i tekstury SlicedEllPack ###
    modSlicedELL = SourceModule(cudaAgregator.getSlicedELLCudaCode(sh_cache_size=threadPerRow*sliceSize, threadPerRow=threadPerRow))
    kernelSlicedELL = modSlicedELL.get_function("SlicedEllpackFormatKernel")
    texrefSlicedELL = modSlicedELL.get_texref("mainVecTexRef")
    
    texrefSlicedELL.set_address(g_wektor, wektor.nbytes)
    texSliced = [texrefSlicedELL]
    ###
    
    ### Mnożenie SlicedEllPack ###     
    for i in range(repeat):
        start.record()
        kernelSlicedELL(vals, \
                            colIdx, \
                            rowLength, \
                            sliceStart, \
                            cuda.Out(wynik), \
                            numRows, \
                            align, \
                            block=block, \
                            grid=grid, \
                            texrefs=texSliced)
        end.record()
        end.synchronize()
        timeList.append(start.time_till(end))
    ###
    
    return (wynik, timeList)    
    
def multiplySertilp(macierz, vector, alignConst, sliceSize, threadPerRow, prefetch = 2, repeat = 1, convertMethod = "new"):    
    if len(vector) != macierz.shape[1]:
        raise ArithmeticError('Length of the vector is not equal to the number of columns of the matrix.')    
    ### Przygotowanie macierzy###
    align = int(ceil((sliceSize*threadPerRow*1.0)/alignConst)*alignConst)
    if convertMethod == 'new':
        mac = transform_to_sertilp(macierz, threads_per_row=threadPerRow, slice_size=sliceSize, prefetch=prefetch, align = alignConst)
        rowLength = mac[2]
#    else: #elif convertMethod == 'old':
#        mac = convert_to_sertilp(macierz, threads_per_row=threadPerRow, slice_size=sliceSize, align=align, prefetch=prefetch)
#        rowLengthTemp = numpy.array([int(ceil((1.0 * i) / (threadPerRow * prefetch))) for i in mac[2]])
#        rowLength = rowLengthTemp
    vals = cuda.to_device(mac[0])
    colIdx = cuda.to_device(mac[1])
    #(int)Math.Ceiling(1.0 * rowLenght[idx] / (threadsPerRow * preFetch))
    
    rowLength = cuda.to_device(rowLength)
    sliceStart = cuda.to_device(mac[3])
    
    wierszeMacierzy, kolumnyMacierzy = macierz.shape
    wektor = vector     
    wynik = numpy.zeros(wierszeMacierzy, dtype=numpy.float32)
    numRows = numpy.int32(wierszeMacierzy)
    ###
    
    ### Przygotowanie stalych wlasciwych ###
    align = numpy.int32(align)
    ###
    
    ### Przygotowanie stałych czasu ###
    timeList = []
    ###    
    
    ### Przygotowanie stałych CUDA ###
    blockSize = threadPerRow * sliceSize;
    gridSize = int(numpy.ceil((wierszeMacierzy*threadPerRow+0.0)/blockSize)) 
    block=(blockSize,1,1)
    grid=(gridSize,1)                    
    g_wektor = cuda.to_device(wektor)
    ###
    
    ### Przygotowanie funkcji i tekstury ###
    mod = SourceModule(cudaAgregator.getSertilpCudaCode(threadPerRow=threadPerRow, sliceSize=sliceSize, prefetch=prefetch))
    kernel = mod.get_function("rbfSERTILP_old")
    texref = mod.get_texref("mainVecTexRef")
    
    texref.set_address(g_wektor, wektor.nbytes)
    tex = [texref]
    ###
    
    ### Mnożenie ###     
    for i in range(repeat):
        start.record()
        kernel(vals, \
                            colIdx, \
                            rowLength, \
                            sliceStart, \
                            cuda.Out(wynik), \
                            numRows, \
                            align, \
                            block=block, \
                            grid=grid, \
                            texrefs=tex)
        end.record()
        end.synchronize()
        timeList.append(start.time_till(end))
    ###
    
    return (wynik, timeList)

def multiplyErtilp(macierz, vector, threadPerRow = 2, prefetch = 2, blockSize = 128, repeat = 1, convertMethod = 'new'):
    if len(vector) != macierz.shape[1]:
        raise ArithmeticError('Length of the vector is not equal to the number of columns of the matrix.')    
#    if convertMethod == 'new':
    if True:
        mac = transform_to_ertilp(macierz, prefetch=prefetch, threads_per_row=threadPerRow)
        rowLength = cuda.to_device(mac[2])
#        rowLength = cuda.to_device(numpy.array([int(ceil((i+0.0)/(threadPerRow*prefetch))) for i in mac[2]]))
#        rowLengthTemp = numpy.ceil(mac[2] / (threadPerRow*prefetch))
#        rowLengthTemp = numpy.array(rowLengthTemp, dtype=numpy.int32)
#        rowLength = cuda.to_device(rowLengthTemp)
#    else:
#        mac = convert_to_ertilp(macierz, threads_per_row=threadPerRow, prefetch=prefetch)
#        rowLength = cuda.to_device(numpy.array([int(ceil((i+0.0)/(threadPerRow*prefetch))) for i in mac[2]]))
    vals = cuda.to_device(mac[0])
    colIdx = cuda.to_device(mac[1])  
    wierszeMacierzy, kolumnyMacierzy = macierz.shape
    wektor = vector    
    wynik = numpy.zeros(wierszeMacierzy, dtype=numpy.float32)
    numRows = numpy.int32(wierszeMacierzy)
    
    ### Przygotowanie stałych czasu ###
    timeList = []
    ###
    
    ### Przygotowanie stałych CUDA ###
    gridSize = int(numpy.ceil((wierszeMacierzy*threadPerRow+0.0)/blockSize))  
    block=(blockSize,1,1)
    grid=(gridSize,1)                    
    g_wektor = cuda.to_device(wektor)
    ###
    
    ### Przygotowanie funkcji i tekstury dla EllPack ###
    mod = SourceModule(cudaAgregator.getErtilpCudaCode(block_sice=blockSize, threadPerRow=threadPerRow, prefetch=prefetch)) 
    kernel = mod.get_function("rbfERTILP")
    texref = mod.get_texref("labelsTexRef")    
    
    texref.set_address(g_wektor, wektor.nbytes)
    tex = [texref]
    ###
    
    
    for i in range(repeat):
        start.record()
        kernel(vals, \
                colIdx, \
                rowLength, \
                cuda.Out(wynik), \
                numRows, \
                block=block, \
                grid=grid, \
                texrefs=tex)
        end.record()
        end.synchronize()
        timeList.append(start.time_till(end))
    
    return (wynik, timeList)

    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:35:44 2014

@author: HP
"""
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import numpy
from math import ceil
 
from matrixFormat import convertToELL, convertToSlicedELL, convertToSertilpELL
import cudaAgregator

import stoper

def multiplyCPU(matrix, repeat = 1):
    wektor = numpy.arange(1, matrix.shape[1]+1, dtype=numpy.float32)
    timer = stoper.Timer()
    
    for i in range(repeat):
        timer.start()
        wynik = matrix.dot(wektor)
        timer.stop()
    
    return (wynik, timer.get_elapsed())

def multiplyELL(macierz, repeat = 1, blockSize = 128): 
    mac = convertToELL(macierz)
    vals = cuda.to_device(mac[0])
    colIdx = cuda.to_device(mac[1])
    rowLength = cuda.to_device(mac[2])
    
    wierszeMacierzy, kolumnyMacierzy = macierz.shape
    wektor = numpy.arange(1, kolumnyMacierzy+1, dtype=numpy.float32)      
    wynik = numpy.zeros(wierszeMacierzy, dtype=numpy.float32)
    numRows = numpy.int32(wierszeMacierzy)
    
    ### Przygotowanie stałych czasu ###
    timer = stoper.Timer()
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
        timer.start()
        kernelELL(vals, \
                colIdx, \
                rowLength, \
                cuda.Out(wynik), \
                numRows, \
                block=block, \
                grid=grid, \
                texrefs=texELL)
        timer.stop()
    
    return (wynik, timer.get_elapsed())
    
def multiplySlicedELL(macierz, alignConst, sliceSize, threadPerRow, repeat = 1):    
    ### Przygotowanie macierzy SlicedEllPack ###
    align = ceil((sliceSize*threadPerRow*1.0)/alignConst)*alignConst
    mac = convertToSlicedELL(macierz, watkiNaWiersz=threadPerRow, sliceSize=sliceSize, align=align)
    vals = cuda.to_device(mac[0])
    colIdx = cuda.to_device(mac[1])
    rowLength = cuda.to_device(mac[2])
    sliceStart = cuda.to_device(mac[3])
    
    wierszeMacierzy, kolumnyMacierzy = macierz.shape
    wektor = numpy.arange(1, kolumnyMacierzy+1, dtype=numpy.float32)      
    wynik = numpy.zeros(wierszeMacierzy, dtype=numpy.float32)
    numRows = numpy.int32(wierszeMacierzy)
    ###
    
    ### Przygotowanie stalych wlasciwych dla SlicedELL
    align = numpy.int32(align)
    sliceSize = numpy.int32(sliceSize)
    ###
    
    ### Przygotowanie stałych czasu ###
    timer = stoper.Timer()
    ###    
    
    ### Przygotowanie stałych CUDA ###
    blockSize = threadPerRow * sliceSize;
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
        timer.start()
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
        timer.stop()
    ###
    
    return (wynik, timer.get_elapsed())    
    
def multiplySertilp(macierz, alignConst, sliceSize, threadPerRow, prefetch = 2, repeat = 1):    
    ### Przygotowanie macierzy###
    align = int(ceil((sliceSize*threadPerRow*1.0)/alignConst)*alignConst)
    mac = convertToSertilpELL(macierz, watkiNaWiersz=threadPerRow, sliceSize=sliceSize, align=align, prefetch=prefetch)
    vals = cuda.to_device(mac[0])
    colIdx = cuda.to_device(mac[1])
    #(int)Math.Ceiling(1.0 * rowLenght[idx] / (threadsPerRow * preFetch))
    rowLengthTemp = numpy.array([int(ceil((1.0 * i) / (threadPerRow * prefetch))) for i in mac[2]])
    rowLength = cuda.to_device(rowLengthTemp)
    sliceStart = cuda.to_device(mac[3])
    
    wierszeMacierzy, kolumnyMacierzy = macierz.shape
    wektor = numpy.arange(1, kolumnyMacierzy+1, dtype=numpy.float32)      
    wynik = numpy.zeros(wierszeMacierzy, dtype=numpy.float32)
    numRows = numpy.int32(wierszeMacierzy)
    ###
    
    ### Przygotowanie stalych wlasciwych ###
    align = numpy.int32(align)
    ###
    
    ### Przygotowanie stałych czasu ###
    timer = stoper.Timer()
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
        timer.start()
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
        timer.stop()
    ###
    
    return (wynik, timer.get_elapsed())

    
    
    
    
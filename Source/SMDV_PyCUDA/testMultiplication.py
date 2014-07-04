# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:42:14 2014

@author: HP
"""

import scipy.io
from matrixUtilites import rowEqualsIgnoreEndZero
from matrixMultiplication import multiplyELL, multiplySlicedELL, multiplyCPU, multiplySertilp
import numpy

def rowEqualsIgnoreEndZeroGetRows(a, b, procentUfnosci = 0.05):
    rows = []
    if len(a) > len(b):
        endMin = len(b)
        objMax = a
    else:
        endMin = len(a)
        objMax = b
    for i in range(endMin):
        average = abs(a[i] + b[i])
        variation = abs(a[i] - b[i])
        if average == 0:
            if round(abs(a[i]), 8) != 0:
                rows.append(i)
        else:
            if (variation / average)*100 > procentUfnosci:
                rows.append(i)
    for i in range(endMin, len(objMax)):
        if objMax[i] != 0:
            return rows.append(i)
    return rows

if __name__ == '__main__':
    ### Stałe programu ###
#    plikMacierzy = 'Macierz_8x8.mtx'
#    plikMacierzy = 'Macierz_9x9.mtx'
#    plikMacierzy = 'Macierz_128x128.mtx'
#    plikMacierzy = 'Macierz_2048x2048_2.mtx'
#    plikMacierzy = 'wbp128.mtx'
#    plikMacierzy = 'wbp256.mtx'
#    plikMacierzy = 'dw8192.mtx'
#    plikMacierzy = 'Macierz_int_100x100.mtx'
#    plikMacierzy = 'Macierz_float_128x128.mtx'
    
#    folderMacierzy = "E:\Slawek\SMVD\SMDV\Macierze\\"
#    folderMacierzy = "E:\Slawek\SMVD\SMDV\Macierze\\wygenerowane\\"
    folderMacierzy = "..\\..\\Matrices\\Generated\\"
    
    blockSize = 128
    sliceSize = 64 # 64 128 
    threadPerRow =2# 2 4 
    alignStala = 32
    prefetch = 2
    powtorzenia = 1
    dokladnoscCzasu = 3
    ###
#    macierz = scipy.io.mmread(folderMacierzy + plikMacierzy)
#    macierz = numpy.array([[0, 0, 3], \
#                           [0, 0, 2], \
#                           [0, 0, 1]])
    
#    resultELLPack = multiplyELL(macierz, blockSize=blockSize, repeat=powtorzenia)
#    print "Czas ELL: " + str(round(resultELLPack[1], dokladnoscCzasu))
#    resultSlicedELLPack = multiplySlicedELL(macierz, alignStala, sliceSize, threadPerRow, repeat=powtorzenia)
#    print "Czas SlicedELL: " + str(round(resultSlicedELLPack[1], dokladnoscCzasu))
#    resultSertilpELLPack = multiplySertilp(macierz, alignStala, sliceSize, threadPerRow, prefetch=prefetch, repeat=powtorzenia)
#    print "Czas SertilpELL: " + str(round(resultSertilpELLPack[1], dokladnoscCzasu))
#    resultCPU = multiplyCPU(macierz, repeat=powtorzenia)
#    print "Czas CPU: " + str(round(resultCPU[1], dokladnoscCzasu))     
#    print "Czy poprawnie ELL - SlicedELL? " + str(rowEqualsIgnoreEndZero(resultELLPack[0], resultSlicedELLPack[0]))
#    print "Czy poprawnie CPU - ELL? " + str(rowEqualsIgnoreEndZero(resultELLPack[0], resultCPU[0]))
#    print "Czy poprawnie ELL - SertilpELL? " + str(rowEqualsIgnoreEndZero(resultELLPack[0], resultSertilpELLPack[0]))
#    print "Czy poprawnie SertilpELL - CPU? " + str(rowEqualsIgnoreEndZero(resultSertilpELLPack[0], resultCPU[0]))
#    rE = resultELLPack[0]
#    rC =  resultCPU[0]
#    rSl = resultSlicedELLPack[0]
#    rS =  resultSertilpELLPack[0]
#    print u"Maksymalny błąd względny ELL - CPU: " + str(max(numpy.abs(rE - rC) / abs(rE + rC))*100)
#    print u"Maksymalny błąd względny SertlipELL - CPU: " + str(max(numpy.abs(rS - rC) / abs(rS + rC))*100)
#    print u"Maksymalny błąd względny SertlipELL - ELL: " + str(max(numpy.abs(rE - rS) / abs(rE + rS))*100)
#    bledneWiersze = rowEqualsIgnoreEndZeroGetRows(rC, rS)
#    print bledneWiersze
    
    for i in range(63, 69):
        plikMacierzy = "Macierz_float_" + str(i) + "x" + str(i) + ".mtx"
        macierz = scipy.io.mmread(folderMacierzy + plikMacierzy)
        resultCPU = multiplyCPU(macierz, repeat=powtorzenia)
        
#        print '========================'
        resultSertilpELLPack = multiplySertilp(macierz, alignConst=alignStala, sliceSize=sliceSize, threadPerRow=threadPerRow, prefetch=prefetch, repeat=powtorzenia)
        if rowEqualsIgnoreEndZero(resultSertilpELLPack[0], resultCPU[0]):
            print plikMacierzy #+ ": " + str(rowEqualsIgnoreEndZeroGetRows(resultSertilpELLPack[0], resultCPU[0]))
#        print resultSertilpELLPack[0]
    print "Skonczylem!"
        
    

    
        
    
    
    
    
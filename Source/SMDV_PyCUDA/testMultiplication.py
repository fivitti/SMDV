# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:42:14 2014

@author: HP
"""

import scipy.io
from matrixUtilites import rowEqualsIgnoreEndZero
from matrixMultiplication import multiplyELL, multiplySlicedELL, multiplyCPU, multiplySertilp, multiplyErtilp
from matrixFormat import convertToSertilpELL
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
#    plikMacierzy = 'Macierz_int_9x9.mtx'
#    plikMacierzy = 'Macierz_float_128x128.mtx'
    plikMacierzy = 'Macierz_int_65x65.mtx'
    
#    folderMacierzy = "E:\Slawek\SMVD\SMDV\Macierze\\"
#    folderMacierzy = "E:\Slawek\SMVD\SMDV\Macierze\\wygenerowane\\"
#    folderMacierzy = "..\\..\\Matrices\\Generated\\"
    folderMacierzy = "../../Matrices/Generated/"
    
    blockSize = 32
    sliceSize =64 # 64 128 
    threadPerRow =2# 2 4 
    alignStala = 32
    prefetch = 2
    powtorzenia = 1
    dokladnoscCzasu = 3
    ###
    macierz = scipy.io.mmread(folderMacierzy + plikMacierzy)
##    from math import ceil
##    conv = convertToSertilpELL(macierz, watkiNaWiersz=threadPerRow, sliceSize=sliceSize, align=int(ceil((sliceSize*threadPerRow*1.0)/alignStala)*alignStala), prefetch=prefetch)
##    macierz = numpy.array([[0, 0, 3], \
##                           [0, 0, 2], \
##                           [0, 0, 1]])
#    
#    resultELLPack = multiplyELL(macierz, blockSize=blockSize, repeat=powtorzenia)
#    print "Czas ELL: " + str(round(resultELLPack[1], dokladnoscCzasu))
#    resultSlicedELLPack = multiplySlicedELL(macierz, alignStala, sliceSize, threadPerRow, repeat=powtorzenia)
#    print "Czas SlicedELL: " + str(round(resultSlicedELLPack[1], dokladnoscCzasu))
#    resultSertilpELLPack = multiplySertilp(macierz, alignStala, sliceSize, threadPerRow, prefetch=prefetch, repeat=powtorzenia)
#    print "Czas SertilpELL: " + str(round(resultSertilpELLPack[1], dokladnoscCzasu))
    resultCPU = multiplyCPU(macierz, repeat=powtorzenia)
    resultErtilp = multiplyErtilp(macierz, threadPerRow, prefetch, blockSize=blockSize, repeat=powtorzenia)
#    print "Czas CPU: " + str(round(resultCPU[1], dokladnoscCzasu))     
#    print "Czy poprawnie ELL - SlicedELL? " + str(rowEqualsIgnoreEndZero(resultELLPack[0], resultSlicedELLPack[0]))
#    print "Czy poprawnie CPU - ELL? " + str(rowEqualsIgnoreEndZero(resultELLPack[0], resultCPU[0]))
#    print "Czy poprawnie ELL - SertilpELL? " + str(rowEqualsIgnoreEndZero(resultELLPack[0], resultSertilpELLPack[0]))
#    print "Czy poprawnie SertilpELL - CPU? " + str(rowEqualsIgnoreEndZero(resultSertilpELLPack[0], resultCPU[0]))
#    rE = resultELLPack[0]
    rC =  resultCPU[0]
#    rSl = resultSlicedELLPack[0]
#    rS =  resultSertilpELLPack[0]
    rEr =resultErtilp[0]
#    print u"Maksymalny błąd względny ELL - CPU: " + str(max(numpy.abs(rE - rC) / abs(rE + rC))*100)
#    print u"Maksymalny błąd względny SertlipELL - CPU: " + str(max(numpy.abs(rS - rC) / abs(rS + rC))*100)
#    print u"Maksymalny błąd względny SertlipELL - ELL: " + str(max(numpy.abs(rE - rS) / abs(rE + rS))*100)
    bledneWiersze = rowEqualsIgnoreEndZeroGetRows(rC, rEr)
    print bledneWiersze
    
#    for i in range(32, 68):
#        blad = False
#        log = ''
#        plikMacierzy = "Macierz_int_" + str(i) + "x" + str(i) + ".mtx"
##        plikMacierzy = "Macierz_int_9x9.mtx"
#        macierz = scipy.io.mmread(folderMacierzy + plikMacierzy)
#        resultCPU = multiplyCPU(macierz, repeat=powtorzenia)
##        resultELLPack = multiplyELL(macierz, blockSize=blockSize, repeat=powtorzenia)
##        resultSlicedELLPack = multiplySlicedELL(macierz, alignStala, sliceSize, threadPerRow, repeat=powtorzenia)        
##        resultSertilpELLPack = multiplySertilp(macierz, alignConst=alignStala, sliceSize=sliceSize, threadPerRow=threadPerRow, prefetch=prefetch, repeat=powtorzenia)
#        resultErtilp = multiplyErtilp(macierz,threadPerRow=threadPerRow, prefetch=prefetch, blockSize=blockSize, repeat=powtorzenia)  
##        if not rowEqualsIgnoreEndZero(resultSertilpELLPack[0], resultCPU[0]):
###            row = rowEqualsIgnoreEndZeroGetRows(resultSertilpELLPack[0], resultCPU[0])
###            print plikMacierzy + " " + str(row)
##            blad = True
##            log += " Sertilp"
##        if not rowEqualsIgnoreEndZero(resultSlicedELLPack[0], resultCPU[0]):
##            blad = True
##            log += " Sliced" 
##        if not rowEqualsIgnoreEndZero(resultELLPack[0], resultCPU[0]):
##            blad = True
##            log += " ELLPack"
##        if not rowEqualsIgnoreEndZero(resultErtilp[0], resultCPU[0]):
##            row = rowEqualsIgnoreEndZeroGetRows(resultErtilp[0], resultCPU[0])
##            print plikMacierzy + " " + str(row)
##            blad = True
##            log += " Ertilp"
##        if blad:
##            print plikMacierzy + log
#            
#        if rowEqualsIgnoreEndZero(resultErtilp[0], resultCPU[0]):
#            print plikMacierzy
            
    print "Skonczylem!"
        
    

    
        
    
    
    
    
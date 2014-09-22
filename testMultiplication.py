# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:42:14 2014

@author: HP
"""

import scipy.io
from matrixUtilites import resultEquals, stringVector
from matrixMultiplication import multiplyCPU, multiplyErtilp
import numpy

if __name__ == '__main__':
    ### Stałe programu ###
    plikMacierzy = 'Macierz_int_65x65.mtx'
#    plikMacierzy = 'Macierz_int_66x66.mtx'
#    plikMacierzy = 'Macierz_int_67x67.mtx'
    plikWektora = 'Vector_65.npy'
#    plikWektora = 'Vector_66.npy'
#    plikWektora = 'Vector_67.npy'
    
    folderMacierzy = "Data/Generated/"
    
    blockSize = 32
    sliceSize =64 # 64 128 
    threadPerRow =2# 2 4 
    alignStala = 32
    prefetch = 2
    powtorzenia = 1
    wspolczynnikUfnosci = 0.01
    ###
    macierz = scipy.io.mmread(folderMacierzy + plikMacierzy)
#    wektor = numpy.load(plikWektora)
    wektor = numpy.arange(1, macierz.shape[1]+1, dtype=numpy.float32)

    resultCPU = multiplyCPU(macierz, vector=wektor, repeat=powtorzenia)
    resultErtilp = multiplyErtilp(macierz, vector=wektor, threadPerRow=threadPerRow, prefetch=prefetch, blockSize=blockSize, repeat=powtorzenia)
    rC =  resultCPU[0]
    rEr =resultErtilp[0]
    bledneWiersze = resultEquals(rC, rEr, wspolczynnikUfnosci)
    print stringVector(bledneWiersze)
            
    print "Skonczylem!"
        
    

    
        
    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 05 13:31:04 2014

@author: HP
"""

import scipy.io
from matrixFormat import *

def convertToSertilpELLBezAlign(macierzDoKonwersji, array = True, watkiNaWiersz = 2, sliceSize = 2, align=8, prefetch = 2):
    u'''
    Metoda przekształca numpy.array w macierz w formacie SertilpELLpack.
    
    Jeżeli parametr array = True to metoda zwraca krotkę, której pierwszym elementem jest numpy.array typu float32 zawierająca niezerowe wartości,
    drugim numpy.array typu int32 zawierająca indeksy kolumn, gdzie znajdują się niezerowe wartości,
    trzecim numpy.array typu int32 zawierająca długości poszczególnych wierszy, czwartym numpy.array typu int 32 zawierająca
    początkowe indeksy każdej z grup na wątek w wartościach i kolumnach.
    
    Jeżeli parametr array = False to metoda zwraca krotkę, której pierwszym elementem jest lista zawierająca niezerowe wartości,
    drugim lista zawierająca indeksy kolumn, gdzie znajdują się niezerowe wartości,
    trzecim lista zawierająca długości poszczególnych wierszy.
    
    Parametr "watkiNaWiersz" określa ile wątków będzie przetwarzało jeden wiersz.
    
    Paramentr "sliceSize" określa ile wierszy będzie składało się na jeden slice.
    
    return (vals, colIdx, rowLength, sliceStart)
    '''
    macierz = preconvertToELL(macierzDoKonwersji)
    reshapeDoWielokrotnosciELL(macierz, sliceSize)
    wartosci = []
    indeksyKolumn = []
    dlugosciWierszy = macierz[2]
    sliceStart = [0, ]
    
    for grupaWierszy in grouped(macierz[0], sliceSize):
       normalizujDlugosci(grupaWierszy, watkiNaWiersz*prefetch)
       wartosci.extend(kolumnyDoListy(grupaWierszy, watkiNaWiersz))
       sliceStart.append(len(wartosci))
    for grupaWierszy in grouped(macierz[1], sliceSize):
        normalizujDlugosci(grupaWierszy, watkiNaWiersz*prefetch)
        indeksyKolumn.extend(kolumnyDoListy(grupaWierszy, watkiNaWiersz))
        
    return (wartosci, indeksyKolumn, dlugosciWierszy, sliceStart)

if __name__ == '__main__':
    plikMacierzy = 'Macierz_int_67x67.mtx'
    folderMacierzy = "..\\..\\Matrices\\Generated\\"
    
    blockSize = 128
    sliceSize = 2 # 64 128 
    threadPerRow =2# 2 4 
    alignStala = 32
    prefetch = 2
    powtorzenia = 1
    dokladnoscCzasu = 3
    
#    macierz = scipy.io.mmread(folderMacierzy + plikMacierzy)
#    m = convertToSertilpELLBezAlign(macierz, watkiNaWiersz=threadPerRow, sliceSize=sliceSize, align=alignStala, prefetch=prefetch)
#    print m
#    ustawAlign(m, alignStala)
    align = ceil((sliceSize*threadPerRow*1.0)/alignStala)*alignStala
    print align
    
    
    
    
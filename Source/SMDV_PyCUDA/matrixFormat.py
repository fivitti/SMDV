# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 20:16:06 2014

@author: HP
"""
import numpy
from itertools import izip
import scipy.sparse

def normalizujDlugosci(listaList, podstawaWielokrotnosci = 1):
    u'''
    Normalizuje długość wszystkich list przekazanych jako argument.
    
    Metoda wyszukuje najdłuższą listę w zestawie. Następnie zwiększa długość 
    wszystkich pozostałych, aby były jej równe. W tym celu dopisuje do nich
    na koniec zera.

    Metoda pracuje "w miejscu". Nie zwraca wyniku.
    '''
    dlugosc = 0
    for l in listaList:
        if len(l) > dlugosc:
            dlugosc = len(l)
            
    reszta = dlugosc % podstawaWielokrotnosci
    if reszta == 0:
        pass
    else:
        dlugosc += podstawaWielokrotnosci - reszta
    
    for l in listaList:
        while len(l) < dlugosc:
            l.append(0)
                
def kolumnyDoListy(listaList, grupuj = 1):
    u'''
    Metoda przekształca macierz przekazaną jako listę list w płaską listę
    sczytując wartości macierzy kolumnami.
    Metoda przyjmuje parametr "grupuj", który określa ile elementów sczytać przy
    każdym przejściu.
    '''
    wynik = []
    count = 0 
    while count < len(listaList):
        count = 0
        for w in listaList:
            for element in range(grupuj):
                if len(w) > 0:
                    wynik.append(w[0])
                    w.remove(w[0])
                else:
                    count += 1
    return wynik


def reshapeDoWielokrotnosciELL(macierzDoRozszerzenia, podstawaWielokrotnosciWierszy):
    u'''
    Metoda rozszerza macierz ELL. Dodaje do przekazanej metody zerowe wiersze.
    Dodawanych jest tyle wierszy, aby ich ilość była wielokrotności przekazanej liczby.
    
    Metoda działa w miejscu, nadpisuje przekazaną macierz.
    '''
    wiersze = len(macierzDoRozszerzenia[2])  
#    doDodania = ((wiersze / podstawaWielokrotnosciWierszy) + 1) * (podstawaWielokrotnosciWierszy - wiersze)
    modulo = wiersze % podstawaWielokrotnosciWierszy
    if modulo == 0:
        return
    else:
        doDodania = podstawaWielokrotnosciWierszy - modulo
    for i in range(doDodania):
        macierzDoRozszerzenia[0].append([])
        macierzDoRozszerzenia[1].append([])
        macierzDoRozszerzenia[2].append(0)

def ustawAlign(macierzSlicedELL, align=64):
    u'''
    Metoda ustawia odpowiedni 'align' dla macierzy go nie posiadającej. Macierz wejściowa musi
    być w formacie: (vals, colIdx, rowLength, sliceStart). Jeżeli 'align' jest mniejszy
    niż rozpiętość  obsługiwana przez jeden wątek metoda nie wprowadza zmian.
    
    Metoda działa w miejscu.
    '''
    licznik = 0
    koniec = macierzSlicedELL[3][licznik]
    wiersze = len(macierzSlicedELL[3]) - 1
    przesuniecie = 0
    while (wiersze > licznik):
        licznik += 1
        macierzSlicedELL[3][licznik] += przesuniecie
        start = macierzSlicedELL[3][licznik-1]
        koniec = macierzSlicedELL[3][licznik]
        roznica = align - (koniec - start)
        if roznica < 0:
            continue
        przesuniecie += roznica
        for i in range(roznica):
            macierzSlicedELL[0].insert(koniec, 0)
            macierzSlicedELL[1].insert(koniec, 0)
            macierzSlicedELL[3][licznik] += 1
            
            
        
        
def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return izip(*[iter(iterable)]*n)
    
def preconvertToELL(macierzDoKonwersji):
    '''    
    Metoda wykonuje początkowe czynności do przekonwertowania macierzy na format ELL.
    
    Zwraca listę zawierającą wartości w poszczególnych wierszach pozbawione zer,
    listę indeksów kolumn zawierających niezerowe wartości,
    a także listę zawierającą liczbę niezerowych wartości w każdym wierszu.
    
    return (vals, indexCols, nonZeroVal)
    '''
    try:
        macierzDoKonwersji = macierzDoKonwersji.tocsr()
    except AttributeError:
        macierzDoKonwersji = scipy.sparse.csr_matrix(macierzDoKonwersji)
        
    dlugosciWierszy = []
    licznikDlugosciWiersza = 0
    wartosciSurowe = []
    kolumnySurowe = []
    for indWiersza in xrange(macierzDoKonwersji.shape[0]):
        wiersz = macierzDoKonwersji.getrow(indWiersza)
        wartosciWiersza = wiersz.data.tolist()
        indeksyWartosciWiersza = wiersz.indices.tolist()
        licznikDlugosciWiersza = len(wartosciWiersza)

        wartosciSurowe.append(wartosciWiersza)
        kolumnySurowe.append(indeksyWartosciWiersza)
        dlugosciWierszy.append(licznikDlugosciWiersza)
    return (wartosciSurowe, kolumnySurowe, dlugosciWierszy)
            

def convertToELL(macierzDoKonwersji, array = True):
    '''
    Metoda przekształca numpy.array w macierz w formacie ELLpack.
    
    Jeżeli parametr array = True to metoda zwraca krotkę, ktrórej pierwszym elementem jest numpy.array typu float32 zawierająca niezerowe wartości,
    drugim numpy.array typu int32 zawierająca indeksy kolumn, gdzie znajdują się niezerowe wartości,
    trzecim numpy.array typu int32 zawierająca długości poszczególnych wierszy.
    
    Jeżeli parametr array = False to metoda zwraca krotkę, której pierwszym elementem jest lista zawierająca niezerowe wartości,
    drugim lista zawierająca indeksy kolumn, gdzie znajdują się niezerowe wartości,
    trzecim lista zawierająca długości poszczególnych wierszy.
    
    return (vals, colIdx, rowLength)
    '''
    wartosci = []
    indeksyKolumn = []
    mdp = preconvertToELL(macierzDoKonwersji)
    wartosciSurowe = mdp[0]
    kolumnySurowe = mdp[1]
    dlugosciWierszy = mdp[2]
        
    normalizujDlugosci(wartosciSurowe)
    normalizujDlugosci(kolumnySurowe)
    
    wartosci = kolumnyDoListy(wartosciSurowe)
    indeksyKolumn = kolumnyDoListy(kolumnySurowe)
    
    if array == True:
        return (numpy.array(wartosci, dtype=numpy.float32), \
                numpy.array(indeksyKolumn, dtype=numpy.int32), \
                numpy.array(dlugosciWierszy, dtype=numpy.int32))
    else:
        return (wartosci, indeksyKolumn, dlugosciWierszy)
                    
def convertToSlicedELL(macierzDoKonwersji, array = True, watkiNaWiersz = 2, sliceSize = 2, align=8):
    u'''
    Metoda przekształca numpy.array w macierz w formacie SlicedELLpack.
    
    Jeżeli parametr array = True to metoda zwraca krotkę, ktrórej pierwszym elementem jest numpy.array typu float32 zawierająca niezerowe wartości,
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
       normalizujDlugosci(grupaWierszy, watkiNaWiersz)
       wartosci.extend(kolumnyDoListy(grupaWierszy, watkiNaWiersz))
       sliceStart.append(len(wartosci))
    for grupaWierszy in grouped(macierz[1], sliceSize):
        normalizujDlugosci(grupaWierszy, watkiNaWiersz)
        indeksyKolumn.extend(kolumnyDoListy(grupaWierszy, watkiNaWiersz))
        
    ustawAlign((wartosci, indeksyKolumn, dlugosciWierszy, sliceStart), align)
   
    if array == True:
        return (numpy.array(wartosci, dtype=numpy.float32), \
                numpy.array(indeksyKolumn, dtype=numpy.int32), \
                numpy.array(dlugosciWierszy, dtype=numpy.int32), \
                numpy.array(sliceStart, dtype=numpy.int32))
    else:
        return (wartosci, indeksyKolumn, dlugosciWierszy, sliceStart)
        
def convertToSertilpELL(macierzDoKonwersji, array = True, watkiNaWiersz = 2, sliceSize = 2, align=8, prefetch = 2):
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
        
    ustawAlign((wartosci, indeksyKolumn, dlugosciWierszy, sliceStart), align)
   
    if array == True:
        return (numpy.array(wartosci, dtype=numpy.float32), \
                numpy.array(indeksyKolumn, dtype=numpy.int32), \
                numpy.array(dlugosciWierszy, dtype=numpy.int32), \
                numpy.array(sliceStart, dtype=numpy.int32))
    else:
        return (wartosci, indeksyKolumn, dlugosciWierszy, sliceStart)

def transformToSERTILP(matrix, threadsPerRow, sliceSize, preFetch, alignParam = 64, array = True):
    try:
        matrix = matrix.tocsr()
    except AttributeError:
        matrix = scipy.sparse.csr_matrix(matrix)
    from math import ceil
    align = int(ceil(1.0 * sliceSize * threadsPerRow / alignParam) * alignParam)
    numRows = matrix.shape[0]
    numSlices = int(ceil((numRows + 0.0) / sliceSize))
    rowLength = [0, ] * numRows
    sliceStart = [0, ] * (numSlices + 1)
    sliceMax = [0, ] * numSlices
    sliceNr = 0
    
    for i in range(numSlices):
        sliceMax[i] = -1
        idx = -1
        for j in range(sliceSize):
            idx = j + i * sliceSize
            if idx < numRows:
                rowLength[idx] = matrix.getrow(idx).getnnz()
                if sliceMax[i] < rowLength[idx]:
                    sliceMax[i] = rowLength[idx]
                rowLength[idx] = int(ceil(1.0 * rowLength[idx] / (threadsPerRow * preFetch)))
        sliceStart[i+1] = sliceStart[i] + int(ceil(1.0*sliceMax[i] / (preFetch * threadsPerRow)) * preFetch * align)
    nnzEl = sliceStart[numSlices]
    vecCols = [0,] * nnzEl
    vecVals = [0,] * nnzEl
    sliceNr = 0
    rowInSlice = 0
    for i in range(numRows):
        sliceNr = i / sliceSize
        rowInSlice = i % sliceSize
        vec = matrix.getrow(i)
        
        threadNr = -1
        value = 0
        col = -1
        
        rowSlice = -1
        for k in range(vec.getnnz()):
            threadNr = k % threadsPerRow
            rowSlice = k / threadsPerRow
            value = vec.data[k]
            col = vec.indices[k]
            idx = sliceStart[sliceNr] + align * rowSlice + rowInSlice * threadsPerRow + threadNr
            
            vecVals[idx] = value
            vecCols[idx] = col
            
    if array == True:
        return (numpy.array(vecVals, dtype=numpy.float32), \
            numpy.array(vecCols, dtype=numpy.int32), \
            numpy.array(rowLength, dtype=numpy.int32), \
            numpy.array(sliceStart, dtype=numpy.int32))
    else:
        return (vecVals, vecCols, rowLength, sliceStart)
        
def convertToErtilp(macierzDoKonwersji, threadPerRow, prefetch, array = True):
    '''    
    Metoda przekształca numpy.array w macierz w formacie Ertilp.
    
    Jeżeli parametr array = True to metoda zwraca krotkę, ktrórej pierwszym elementem jest numpy.array typu float32 zawierająca niezerowe wartości,
    drugim numpy.array typu int32 zawierająca indeksy kolumn, gdzie znajdują się niezerowe wartości,
    trzecim numpy.array typu int32 zawierająca długości poszczególnych wierszy.
    
    Jeżeli parametr array = False to metoda zwraca krotkę, której pierwszym elementem jest lista zawierająca niezerowe wartości,
    drugim lista zawierająca indeksy kolumn, gdzie znajdują się niezerowe wartości,
    trzecim lista zawierająca długości poszczególnych wierszy.
    
    return (vals, colIdx, rowLength)
    '''
    wartosci = []
    indeksyKolumn = []
    mdp = preconvertToELL(macierzDoKonwersji)
    wartosciSurowe = mdp[0]
    kolumnySurowe = mdp[1]
    dlugosciWierszy = mdp[2]
        
    normalizujDlugosci(wartosciSurowe, threadPerRow*prefetch)
    normalizujDlugosci(kolumnySurowe, threadPerRow*prefetch)
    
    wartosci = kolumnyDoListy(wartosciSurowe, threadPerRow)
    indeksyKolumn = kolumnyDoListy(kolumnySurowe, threadPerRow)
    
    if array == True:
        return (numpy.array(wartosci, dtype=numpy.float32), \
                numpy.array(indeksyKolumn, dtype=numpy.int32), \
                numpy.array(dlugosciWierszy, dtype=numpy.int32))
    else:
        return (wartosci, indeksyKolumn, dlugosciWierszy)
        
def transformToERTILPFormat(matrix, align, ThreadsPerRow, array = True):
    '''
        align - ThreadsPerRow * prefetch
    '''
    try:
        matrix = matrix.tocsr()
    except AttributeError:
        matrix = scipy.sparse.csr_matrix(matrix)
    from math import ceil#, mean
    
    maxEl = 1
    maxEl = max([m.getnnz() for m in matrix])
    rest = maxEl % align
    
    if rest != 0:
        maxEl = maxEl + align - rest
    
#    avgEl = mean([m.getnnz() for m in matrix])
    numRows = matrix.shape[0]
    vecVals = [0, ] * (numRows * maxEl)
    vecCols = [0, ] * (numRows * maxEl)
    rowLength = [0, ] * numRows
    
    for i in range(numRows):
        vec = matrix.getrow(i)
        for j in range(vec.getnnz()):
            k = j / ThreadsPerRow
            t = j % ThreadsPerRow
            vecVals[k * numRows * ThreadsPerRow + i * ThreadsPerRow + t] = vec.data[j]
            vecCols[k * numRows * ThreadsPerRow + i * ThreadsPerRow + t] = vec.indices[j]
        rowLength[i] = int(ceil((vec.getnnz() + 0.0) / align))

    if array == True:
        return (numpy.array(vecVals, dtype=numpy.float32), \
            numpy.array(vecCols, dtype=numpy.int32), \
            numpy.array(rowLength, dtype=numpy.int32))
    else:
        return (vecVals, vecCols, rowLength)
        
if __name__ == "__main__":
    A = numpy.array([[3, 0, 5, 0, 2],
                     [0, 1, 0, 3, 2],
                     [0, 2, 3, 0, 0],
                     [0, 0, 1, 0, 9]])
    B = numpy.array([[1, 0, 0, 3, 4],
                     [0, 0, 12, 13, 0],
                     [0, 21, 0, 0, 0],
                     [30, 31, 0, 0, 0],
                     [40, 41, 42, 43, 0],
                     [0, 0, 0, 0, 54]])
    C = numpy.array([[1,0,2,0,3,0], 
                     [4,0,5,0,0,0],
                     [0,0,0,6,7,0],
                     [0,0,0,0,0,8],
                     [21,0,22,0,23,0], 
                     [24,0,25,0,0,0],
                     [0,0,0,26,27,0],
                     [0,0,0,0,0,28]])
    D = numpy.array([[0, 1, 0, 2, 0, 0],
                      [1, 0, 0, 0, 0, 0],
                      [1, 2, 0, 3, 4, 5],
                      [0, 1, 0, 0, 0, 2],
                      [1, 2, 0, 0, 0, 0],
                      [0, 1, 2, 0, 3, 0]])     
    E = numpy.array([
                     [0, 0, 0, 0, 0, 0, 0, 0, 18],
                     [0, 30, 0, 0, 0, 0, 49, 0, 0],
                     [0, 0, 30, 0, 0, 0, 0, 0, 0],
                     [0, 35, 85, 0, 27, 0, 45, 0, 0],
                     [52, 42, 0, 0, 0, 0, 44, 14, 69],
                     [0, 0, 0, 0, 0, 0, 0, 14, 0],
                     [41, 0, 0, 0, 0, 37, 4, 0, 70],
                     [0, 0, 0, 79, 11, 0, 5, 0, 0],
                     [0, 24, 40, 0, 0, 83, 30, 0, 0]])
                     
#    macierze = [A, B, C]
    macierze = [E, ] 
    sliceSize = 4 # 64 128 
    threadPerRow = 2# 2 4 
    
    prefetch = 2
    alignStala = prefetch*threadPerRow
    
    from matrixUtilites import stringListInList
    
    for macierz in macierze:
#        mELL = convertToELL(macierz, array=False)
#        mSlicedELL = convertToSlicedELL(macierz, array=False, watkiNaWiersz=threadPerRow, sliceSize=sliceSize, align=alignStala)
#        mSertilpELL = convertToSertilpELL(macierz, array=False, watkiNaWiersz=threadPerRow, sliceSize=sliceSize, align=alignStala, prefetch=prefetch)
#        mSertilpELLTransform = transformToSERTILP(macierz, threadsPerRow=threadPerRow, sliceSize=sliceSize, preFetch=prefetch, alignParam=alignStala)
        eT = transformToERTILPFormat(macierz, align=alignStala, ThreadsPerRow=threadPerRow)
        eC = convertToErtilp(macierz, threadPerRow=threadPerRow, prefetch=prefetch)        
        print "Macierz:\n" + str(macierz) 
#        print "ELL:\n" + stringListInList(mELL)
#        print "SlicedELL:\n" + stringListInList(mSlicedELL)
#        print "SertilpELL:\n" + stringListInList(mSertilpELL)
#        print "SertilpELL:\n" + stringListInList(mSertilpELLTransform)

        
        

    
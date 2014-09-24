# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 20:16:06 2014

@author: Sławomir Figiel
"""
import numpy
import scipy.sparse
from listutilites import normalize_length, columns_to_list, grouped

def reshape_to_multiple_ell(matrix_to_extension, multiple_row):
    '''
    Method extends matrix preconvert Ellpack. Adds empty list to the 
    list of rows and columns and zero to the list length of rows.
    Adds many elements that the total number in each list
    (rows, columns, length of lines) be a multiple parameter multiple_row.

    Parameters
    ==========
    matrix_to_extension : list of list
        Represents matrix preconvert Ellpack. Should have three parts: 
        list of values​​, list of columns and list of row length.
        List of values and columns are list of list with rows. List of
        row length is list of integers.
    multiple_row : integer > 0
        The length of each list the resulting matrix is a multiple of 
        this number.
        
    Returns
    =======
    Nothing. Method works in place.
    
    Notes
    =====
    If the length of the lists are already multiple multiple_row method
    does not change anything.
    
    Examples
    ========
    >>> pre_ell = [[[1], [3, 1], [7]], [[1], [0, 2], [2]], [1, 2, 1]]
    >>> reshape_to_multiple_ell(pre_ell, 4)
    >>> pre_ell
        [[[1], [3, 1], [7], []], [[1], [0, 2], [2], []], [1, 2, 1, 0]]
    '''
    rows = len(matrix_to_extension[2])  
    modulo = rows % multiple_row
    if modulo == 0:
        return
    to_add = multiple_row - modulo
    matrix_to_extension[0].extend([ [] ]*to_add)
    matrix_to_extension[1].extend([ [] ]*to_add)
    matrix_to_extension[2].extend([0,]*to_add)

def set_align(matrix_sliced_ell, align=64):
    u'''
    Metoda ustawia odpowiedni 'align' dla macierzy go nie posiadającej. Macierz wejściowa musi
    być w formacie: (vals, colIdx, rowLength, sliceStart). Jeżeli 'align' jest mniejszy
    niż rozpiętość  obsługiwana przez jeden wątek metoda nie wprowadza zmian.
    
    Metoda działa w miejscu.
    '''
    counter = 0
    end = matrix_sliced_ell[3][counter]
    rows = len(matrix_sliced_ell[3]) - 1
    shift = 0
    while (rows > counter):
        counter += 1
        matrix_sliced_ell[3][counter] += shift
        start = matrix_sliced_ell[3][counter-1]
        end = matrix_sliced_ell[3][counter]
        different = align - (end - start)
        if different < 0:
            continue
        shift += different
        for i in range(different):
            matrix_sliced_ell[0].insert(end, 0)
            matrix_sliced_ell[1].insert(end, 0)
            matrix_sliced_ell[3][counter] += 1     
    
def preconvertToELL(macierzDoKonwersji):
    '''    
    Metoda wykonuje początkowe czynności do przekonwertowania macierzy na format ELL.
    
    Zwraca listę zawierającą wartości w poszczególnych wierszach pozbawione zer,
    listę indeksów kolumn zawierających niezerowe wartości,
    a także listę zawierającą liczbę niezerowych wartości w każdym wierszu.
    
    return (vals, indexCols, nonZeroVal)
    '''
    macierzDoKonwersji = transformToScipyCsr(macierzDoKonwersji)
        
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
        
    wartosciSurowe = normalize_length(wartosciSurowe)
    kolumnySurowe = normalize_length(kolumnySurowe)
    
    wartosci = columns_to_list(wartosciSurowe)
    indeksyKolumn = columns_to_list(kolumnySurowe)
    
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
    reshape_to_multiple_ell(macierz, sliceSize)
    wartosci = []
    indeksyKolumn = []
    dlugosciWierszy = macierz[2]
    sliceStart = [0, ]
    
    for grupaWierszy in grouped(macierz[0], sliceSize):
       grupaWierszy = normalize_length(grupaWierszy, watkiNaWiersz)
       wartosci.extend(columns_to_list(grupaWierszy, watkiNaWiersz))
       sliceStart.append(len(wartosci))
    for grupaWierszy in grouped(macierz[1], sliceSize):
        grupaWierszy = normalize_length(grupaWierszy, watkiNaWiersz)
        indeksyKolumn.extend(columns_to_list(grupaWierszy, watkiNaWiersz))
        
    set_align((wartosci, indeksyKolumn, dlugosciWierszy, sliceStart), align)
   
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
    reshape_to_multiple_ell(macierz, sliceSize)
    wartosci = []
    indeksyKolumn = []
    dlugosciWierszy = macierz[2]
    sliceStart = [0, ]
    
    for grupaWierszy in grouped(macierz[0], sliceSize):
       grupaWierszy = normalize_length(grupaWierszy, watkiNaWiersz*prefetch)
       wartosci.extend(columns_to_list(grupaWierszy, watkiNaWiersz))
       sliceStart.append(len(wartosci))
    for grupaWierszy in grouped(macierz[1], sliceSize):
        grupaWierszy = normalize_length(grupaWierszy, watkiNaWiersz*prefetch)
        indeksyKolumn.extend(columns_to_list(grupaWierszy, watkiNaWiersz))
        
    set_align((wartosci, indeksyKolumn, dlugosciWierszy, sliceStart), align)
   
    if array == True:
        return (numpy.array(wartosci, dtype=numpy.float32), \
                numpy.array(indeksyKolumn, dtype=numpy.int32), \
                numpy.array(dlugosciWierszy, dtype=numpy.int32), \
                numpy.array(sliceStart, dtype=numpy.int32))
    else:
        return (wartosci, indeksyKolumn, dlugosciWierszy, sliceStart)

def transformToSERTILP(matrix, threadsPerRow, sliceSize, preFetch, alignParam = 64, array = True):
    matrix = transformToScipyCsr(matrix)
    
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
        
    wartosciSurowe = normalize_length(wartosciSurowe, threadPerRow*prefetch)
    kolumnySurowe = normalize_length(kolumnySurowe, threadPerRow*prefetch)
    
    wartosci = columns_to_list(wartosciSurowe, threadPerRow)
    indeksyKolumn = columns_to_list(kolumnySurowe, threadPerRow)
    
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
    matrix = transformToScipyCsr(matrix)
    
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

def transformToScipyCsr(matrix):
    if scipy.sparse.isspmatrix_csr(matrix):
        return matrix
    elif scipy.sparse.isspmatrix(matrix):
        return matrix.tocsr()
    elif isinstance(matrix, numpy.ndarray):
        return scipy.sparse.csr_matrix(matrix)
    else:
        raise NotImplementedError('This matrix type is not supported. Only support numpy.ndarray and scipy.sparse matrix.')
        
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
                     
        

    
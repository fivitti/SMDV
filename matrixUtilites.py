# -*- coding: utf-8 -*-
"""
Created on Sun May 04 20:45:18 2014

@author: Sławomir Figiel
"""
from random import randint, uniform
import numpy
import time
import os
import scipy.io
import scipy.sparse

def generateMatrixCsr(rows = 5, cols = 5, minimum = 1, maximum = 100, integers = True, mode = 0, percentageOfZeros = 40, precision=6):
    u'''
    Metoda generuje losową macierz o zadanych parametrach.
    
    Opis parametrów:
        cols - cols macierzy,
        rows - rows macierzy,
        minimum - minimalna wartość niezerowLista w macierzy,
        maximum - maksymalna wartość niezerowLista w macierzy,
        integers - jeżeli prawda wylosowane zostaną tylko liczby całkowite
        mode - dla 0 - wyjściowa macierz w formacie CSR, dla 1 - wyjściowa macierz w formacie CSR dense,
                dla 2 - wyjściowa macierz w formacie numpy array
        procent zer - procentowa zawartość zer w macierzy
    '''
    toDraw = int(round(cols*rows - (cols*rows*percentageOfZeros)/100))
    rowList = []
    colList = []
    data = []
    points = set([])
    randMethod = 0
    if integers:
        randMethod = randint
    else:
        randMethod = uniform
    while toDraw > 0:
        row = randint(0, rows-1)
        col = randint(0, cols-1)
        point = (row, col)
        if not point in points:
            points.add(point)
            toDraw -= 1
    for i in points:
        rowList.append(i[0])
        colList.append(i[1])
        value = round(randMethod(minimum, maximum), precision)
        while value == 0:
            value = round(randMethod(minimum, maximum), precision)
        data.append(value)

    matrix = scipy.sparse.csr_matrix( (data,(rowList,colList)), shape=(rows,cols), dtype=numpy.float32 )
        
    if mode == 0:
        return matrix
    elif mode == 1:
        return matrix.todense()
    elif mode == 2:
        return matrix.toarray()
    else:
        return matrix

def generateVector(length=5, minimum=-1, maximum=1, integers=True, percentageOfZeros=40, precision=6, array=True):
    if percentageOfZeros > 100:
        percentageOfZeros = 100
    elif percentageOfZeros < 0:
        percentageOfZeros = 0
    result = []
    zeros = round((length * percentageOfZeros * 1.0) / 100)
    randMethod = randint if integers else uniform
    for i in range(length):
        drawn = round(randMethod(minimum, maximum), precision)
        while drawn == 0:
            drawn = round(randMethod(minimum, maximum), precision)
        result.append(drawn)
    while zeros > 0:
        place = randint(0, length-1)
        if not result[place] == 0:
            result[place] = 0
            zeros -= 1
    if array:
        typeData = numpy.float32
        return numpy.array(result, dtype=typeData)
    else:
        return result
            
def saveVectorToNumpyFile(vector, folder = 'vectors\\', prefix = 'Vector_', extension = '.npy', date=False, length=True, suffix=''):
    filePath = folder
    filePath += prefix
    if length:
        filePath += str(len(vector))
    if date:
        t = time.localtime()
        for i in range(6):
            if t[i] < 10:
                filePath += '0'
            filePath += str(t[i])
    if os.path.exists(filePath+suffix+extension):
        addition = 1
        while os.path.exists(filePath+'_'+str(addition)+suffix+extension):
            addition += 1
        filePath += '_' + str(addition)
    filePath += suffix
    filePath += extension    
    numpy.save(filePath, vector)

def saveMatrixToFile(matrix, folder = 'macierze\\', prefix = 'Macierz_', extension = '.mtx', date=False, dimensions=True, suffix=''):
    u'''
    Zapisuje przekazaną matrix do pliku w formacie MatrixMarket.

    Opis parametrów:
        matrix - matrix do zapisania
        folder - folder zapisu
        prefix - prefix dodany do nazwy
        extension - extension wyjściowego pliku
        date - jeżeli prawdziwe, na końcu nazwy pliku zostanie dodany ciąg w formacie RokMiesiacDzienGodzinaMinutaSekunda
        dimensions - jeżeli prawdziwe, po przedrostku dodany zostanie ciąg zawierający kolejne dimensions matrix oddzielone znakiem 'x'
        suffix - dodatkowy tekst, który zostanie dodany tuż przed rozszerzeniem
        Jeżeli przed zapisem okaże się, że plik o danej nazwie istnieje dodany zostanie na końcu kolejny nieistniejący numer, począwszy od 1.
    '''
    filepath = folder
    filepath += prefix
    if dimensions:
        dimensionsList = []
        for i in matrix.shape:
            dimensionsList.append(str(i))
        filepath += 'x'.join(dimensionsList)
    if date:
        now = time.localtime()
        for i in range(6):
            if now[i] < 10:
                filepath += '0'
            filepath += str(now[i])
    if os.path.exists(filepath+suffix+extension):
        addition = 1
        while os.path.exists(filepath+'_'+str(addition)+suffix+extension):
            addition += 1
        filepath += '_' + str(addition)
    filepath += suffix
    filepath += extension    
    scipy.io.mmwrite(filepath, matrix)
    
def formatItem(left, right, width=40, rowFormat='  {0:<7}{1:>9}'):
    from textwrap import fill
    wrapped = fill(right, width=width, subsequent_indent=' '*15)
    return rowFormat.format(left, wrapped)
def stringVector(vector, withoutZeros = False, valueFormat="%.15g", width=40, rowFormat='  {0:<7}{1:>9}'):
    result = []
    for idx, val in enumerate(vector):
        if withoutZeros and not val:
            continue
        result.append(formatItem('(%s)' % str(idx),  valueFormat % val, width=width, rowFormat=rowFormat))
    return '\n'.join(result)
def twoColumnString(leftList, rightList, rightWidth=80, rowFormat='  {0:<5}{1:>10}'):
    result = []
    for left, right in zip(leftList, rightList):
        result.append(formatItem('%s' % left,  '%s' % right, width=rightWidth, rowFormat=rowFormat))
    return '\n'.join(result)
    
def getShapeEll(matrixEll):
    rows = len(matrixEll[2])
    cols = max(matrixEll[1])+1
    return (rows, cols)

def getInfoMatrix(matrix):
    shape = matrix.shape
    nnz = matrix.nnz
    sparsing = round( ( (nnz+0.0)/ ( (shape[0] * shape[1]) ) ) * 100, 4)
    return (shape[0], shape[1], nnz, sparsing)
    
def getInfoVector(vector):           
    length = len(vector)
    nnz = sum(1 for i in vector if not i)
    sparsing = round( ( (nnz+0.0)/ ( (length) ) ) * 100, 4) 
    return (length, nnz, sparsing)  
        
def resultEquals(correct, current, confidenceFactor = 0.0005):
    '''
    Return list tuple if errors:
        (index, different, relative error)
    
    If the length of the list correct and current are different additional fields should be zero.
    If not as different is "#". (index, #, #)
    If the array contains floating-point numbers, set the appropriate confidence factor.
    (correct - correct*confidenceFactor : correct + correct*confidenceFactor)
    Returns a list of pairs: the number of fields in the list, and the difference from 
    the correct result [correct - current].
    The third value is the relative error. 
    If the correct value is 0 instead it is given the character '#' (index, difference, #)
    '''
    result = []
    if len(correct) > len(current):
        endMin = len(current)
        objMax = correct
    else:
        endMin = len(correct)
        objMax = current
    for i in range(endMin):
        if correct[i] == 0:
            if round(abs(current[i]), 8) != 0:
                result.append((i, current[i]*(-1), '#'))
        else:
            if (correct[i] > 0 and (current[i] > correct[i]*(1+confidenceFactor) or current[i] < correct[i]*(1-confidenceFactor))) or \
               (correct[i] < 0 and (current[i] < correct[i]*(1+confidenceFactor) or current[i] > correct[i]*(1-confidenceFactor))):
                different = correct[i] - current[i]
                result.append((i, different, different/correct[i]))
    for i in range(endMin, len(objMax)):
        if objMax[i] != 0:
            return result.append((i, '#', '#'))
    return result

def dictVectorPaths(vectorPaths):
    '''
    Method return dict where key is length of vector and value is path to vector.
        dict[len(vector)] = pathVector
    '''
    result = {}
    for path in vectorPaths:
        try:
            vector = numpy.load(path)
        except:
            continue
        result[len(vector)] = path
    return result
  
if __name__ == '__main__':
    pass
    
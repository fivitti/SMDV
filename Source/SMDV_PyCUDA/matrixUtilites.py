# -*- coding: utf-8 -*-
"""
Created on Sun May 04 20:45:18 2014

@author: HP
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
#    if integers:
#        matrix = scipy.sparse.csr_matrix( (data,(rowList,colList)), shape=(cols,rows), dtype=numpy.int32 )
#    else:
#    print rowList, colList, data
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
            
    

def generujMacierz_Normal(szerokosc = 5, wysokosc = 5, minimum = 0, maksimum = 100, calkowite = True, array = False, procentZer = 40):
    macierz = []
    for i in range(wysokosc):
        wiersz = []
        for j in range(szerokosc):
            if calkowite:
                wiersz.append(randint(minimum, maksimum))
            else:
                wiersz.append(uniform(minimum, maksimum))
        macierz.append(wiersz)
        
    if procentZer > 0 and procentZer <= 100:
        zeraDoWylosowania = round((szerokosc*wysokosc*procentZer)/100)
        for i in macierz:
            for j in i:
                if j == 0:
                    if zeraDoWylosowania > 0:
                        zeraDoWylosowania -= 1
                    else:
                        while j == 0:
                            if calkowite:
                                j = randint(minimum, maksimum)
                            else:
                                j = uniform(minimum, maksimum)
        while zeraDoWylosowania > 0:
            x = randint(1, szerokosc) - 1
            y = randint(1, wysokosc) - 1
            if macierz[y][x] == 0:
                continue
            else:
                macierz[y][x] = 0
                zeraDoWylosowania -= 1      
    if array:
        if calkowite:
            return numpy.array(macierz, dtype = numpy.int32)
        else:
            return numpy.array(macierz, dtype = numpy.float32)
    else:      
        return macierz

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

def printListInList(listaList):
    for i in listaList:
        print i
def stringListInList(listaList):
    s = ''
    for i in listaList:
        s += str(i) + "\n"
    return s
    
def getShapeEll(macierz):
    wiersze = len(macierz[2])
    kolumny = max(macierz[1])+1
    return (wiersze, kolumny)
    
def rowEqualsIgnoreEndZero(a, b, procentUfnosci = 0.05):
    '''
    Metoda przyjmuje dwie listy i sprawdza, czy posiadają takie same odpowiadające elementy.
    Zwróci prawdę, gdy listy są identyczne.
    W przypadku, gdy jedna z list jest dłuższa niż długa prawda zostanie zwrócona jeżeli
    nadmiarowe elementy są zerami.
    
    Parametr "procentUfnosci" określa jakim procentem średniej wartości na listach a i b (dla odpowiadających
    sobie indeksów) może być różnica jednej z tych wartości i średniej.
    
    Przykład:
        a = [x_1, x_2, x_3, x_4, ...]
        b = [y_1, y_2, y_3, y_4, ...]
        Jeżeli dla i = 1, 2, 3, 4, ...
            avr_i = | (a_i + _ib) / 2 |
            var_i = | avr_i - a_i |
            par_i = var_i / avr_i = | avr_i - a_i | / | (a_i + b_i) / 2 | =
                  = | (a_i + b_i) / 2 - a_i | / | (a_i + b_i) / 2 | =
                  = | (a_i + b_i - 2 * a_i) / 2 | / | (a_i + b_i) / 2 | =
                  = | b_i - a_i | / | a_i + b_i |
        par_i * 100 > procentUfnosci zwraca fałsz,
        jeżeli nie kontynuuje obliczenia dla kolejnego i.
        W przypadku, gdy średnia będzie równa zero, odchylenie (czyli a_i, lub b_i)
        zostaje zaokrąglone do ósmego miejsca po przecinku. Jeżeli tak zaookrąglona wartość
        nie będzie równa zero to zwrócony zostanie fałsz.
    '''
    
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
                return False
        else:
            if (variation / average)*100 > procentUfnosci:
                return False
    for i in range(endMin, len(objMax)):
        if objMax[i] != 0:
            return False
    return True
    
def resultEquals(correct, current, confidenceFactor = 0.0005):
    '''
    If the length of the list correct and current are different additional fields should be zero.
    If not as different is "#".
    If the array contains floating-point numbers, set the appropriate confidence factor.
    (correct - correct*confidenceFactor : correct + correct*confidenceFactor)
    Returns a list of pairs: the number of fields in the list, and the difference from 
    the correct result [correct - current].
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
                result.append((i, current[i]*(-1)))
        else:
            if (correct[i] > 0 and (current[i] > correct[i]*(1+confidenceFactor) or current[i] < correct[i]*(1-confidenceFactor))) or \
               (correct[i] < 0 and (current[i] < correct[i]*(1+confidenceFactor) or current[i] > correct[i]*(1-confidenceFactor))):
                result.append((i, correct[i] - current[i]))
    for i in range(endMin, len(objMax)):
        if objMax[i] != 0:
            return result.append((i, '#'))
    return result
        
  
if __name__ == '__main__':
#    A = generujMacierz_Csr(szerokosc=10, wysokosc=1, procentZer=80, tryb=0)
#    B = generujMacierz_Normal(szerokosc=10, wysokosc=10, procentZer=80, array=True)
#    print 'Macierz A - CSR:\n' + str(A)
#    print 'Macierz B:\n' + str(B)
#    folder = 'E:\Moje projekty\SMDV\Data\\Generated\\'
#    folder = '../../Data/Generated'
#    wymiary = [5, 6]
#    procentZer = 15
#    for i in wymiary:
#        v = generateVector(length=i, procentageOfZeros=procentZer, integers=False)
#        saveVectorToNumpyFile(v, folder, prefix='Vector_float_', suffix='_'+str(procentZer)+'p')
#    print numpy.load(folder + 'Vector_int_5.npy')

#    A = [3, 2, 1, 0, 3]
#    B = [3, 2, 1, 0, 3, 0, 0, 0, 0, 0]
#    assert rowEqualsIgnoreEndZero(A, B)
           
#    folder =  'E:\\Slawek\\SMVD\\SMDV\\Macierze\\wygenerowane\\'
#    folder = 'E:\Moje projekty\SMDV\Macierze\\wygenerowane\\'
#    wymiary = range(5)
#    for i in wymiary:
#        zapiszMacierzDoPliku(generujMacierz_Csr(szerokosc=i, wysokosc=1, procentZer=70, calkowite = True), przedrostek="Vector_int_", folder=folder)
           
#    procentZer = 75
#    minimum = 1
#    maksimum = 9
#    for i in wymiary:
#        matrix = generujMacierz_Csr(szerokosc=i, wysokosc=i, minimum=minimum, maksimum=maksimum, procentZer=procentZer)
#        zapiszMacierzDoPliku(matrix, folder=folder)
    
#    A = numpy.array([67, 4, -84, -7, 8, 133], dtype=numpy.int32)
#    B = numpy.array([67.0, 4.0, -84.0, -7.0, 8.0, 133.0], dtype=numpy.float32)
    A = generateMatrixCsr(minimum = -10, maximum=10)
    print A

#    print resultEquals(A, B)    
    
        
    
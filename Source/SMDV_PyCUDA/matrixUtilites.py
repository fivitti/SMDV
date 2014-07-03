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

def generujMacierz_Csr(szerokosc = 5, wysokosc = 5, minimum = 1, maksimum = 100, calkowite = True, tryb = 0, procentZer = 40):
    u'''
    Metoda generuje losową macierz o zadanych parametrach.
    
    Opis parametrów:
        szerokosc - szerokosc macierzy,
        wysokosc - wysokosc macierzy,
        minimum - minimalna wartość niezerowa w macierzy,
        maksimum - maksymalna wartość niezerowa w macierzy,
        calkowite - jeżeli prawda wylosowane zostaną tylko liczby całkowite
        tryb - dla 0 - wyjściowa macierz w formacie CSR, dla 1 - wyjściowa macierz w formacie CSR dense,
                dla 2 - wyjściowa macierz w formacie numpy array
        procent zer - procentowa zawartość zer w macierzy
    '''
    doWylosowania = round(szerokosc*wysokosc - (szerokosc*wysokosc*procentZer)/100)
    row = []
    col = []
    data = []
    punkty = set([])
    losujaca = 0
    if calkowite:
        losujaca = randint
    else:
        losujaca = uniform
    while doWylosowania > 0:
        wiersz = randint(0, wysokosc-1)
        kolumna = randint(0, szerokosc-1)
        punkt = (wiersz, kolumna)
        if not punkt in punkty:
            punkty.add(punkt)
            doWylosowania -= 1
    for i in punkty:
        row.append(i[0])
        col.append(i[1])
        wartosc = losujaca(minimum, maksimum)
        while wartosc == 0:
            wartosc = losujaca(minimum, maksimum)
        data.append(wartosc)
        
    if calkowite:
        matrix = scipy.sparse.csr_matrix( (data,(row,col)), shape=(szerokosc,wysokosc), dtype=numpy.int32 )
    else:
        matrix = scipy.sparse.csr_matrix( (data,(row,col)), shape=(szerokosc,wysokosc), dtype=numpy.float32 )
        
    if tryb == 0:
        return matrix
    elif tryb == 1:
        return matrix.todense()
    elif tryb == 2:
        return matrix.toarray()
    else:
        return matrix
        
        
        

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
    
def zapiszMacierzDoPliku(macierz, folder = 'macierze\\', przedrostek = 'Macierz_', rozszerzenie = '.mtx', data=False, wymiary=True, dopisek=''):
    u'''
    Zapisuje przekazaną macierz do pliku w formacie MatrixMarket.

    Opis parametrów:
        macierz - macierz do zapisania
        folder - folder zapisu
        przedrostek - przedrostek dodany do nazwy
        rozszerzenie - rozszerzenie wyjściowego pliku
        data - jeżeli prawdziwe, na końcu nazwy pliku zostanie dodany ciąg w formacie RokMiesiacDzienGodzinaMinutaSekunda
        wymiary - jeżeli prawdziwe, po przedrostku dodany zostanie ciąg zawierający kolejne wymiary macierz oddzielone znakiem 'x'
        dopisek - dodatkowy tekst, który zostanie dodany tuż przed rozszerzeniem
        Jeżeli przed zapisem okaże się, że plik o danej nazwie istnieje dodany zostanie na końcu kolejny nieistniejący numer, począwszy od 1.
    '''
    nazwa = folder
    nazwa += przedrostek
    if wymiary:
        wymiaryLista = []
        for i in macierz.shape:
            wymiaryLista.append(str(i))
        nazwa += 'x'.join(wymiaryLista)
    if data:
        czas = time.localtime()
        for i in range(6):
            if czas[i] < 10:
                nazwa += '0'
            nazwa += str(czas[i])
    if os.path.exists(nazwa+rozszerzenie):
        dodatek = 1
        while os.path.exists(nazwa+'_'+str(dodatek)+rozszerzenie):
            dodatek += 1
        nazwa += '_' + str(dodatek)
    nazwa += dopisek
    nazwa += rozszerzenie    
    scipy.io.mmwrite(nazwa, macierz)

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
        
  
if __name__ == '__main__':
    A = generujMacierz_Csr(szerokosc=10, wysokosc=10, procentZer=80, tryb=0)
    B = generujMacierz_Normal(szerokosc=10, wysokosc=10, procentZer=80, array=True)
    print 'Macierz A - CSR:\n' + str(A)
    print 'Macierz B:\n' + str(B)

    A = [3, 2, 1, 0, 3]
    B = [3, 2, 1, 0, 3, 0, 0, 0, 0, 0]
    assert rowEqualsIgnoreEndZero(A, B)
           
    folder =  'E:\\Slawek\\SMVD\\SMDV\\Macierze\\wygenerowane\\'
#    folder = 'E:\Moje projekty\SMDV\Macierze\\wygenerowane\\'
    wymiary = range(128, 256)
    for i in wymiary:
        zapiszMacierzDoPliku(generujMacierz_Csr(szerokosc=i, wysokosc=i, procentZer=70, calkowite = False), przedrostek="Macierz_float_", folder=folder)
           
#    procentZer = 75
#    minimum = 1
#    maksimum = 9
#    for i in wymiary:
#        matrix = generujMacierz_Csr(szerokosc=i, wysokosc=i, minimum=minimum, maksimum=maksimum, procentZer=procentZer)
#        zapiszMacierzDoPliku(matrix, folder=folder)
    
    
    
        
    
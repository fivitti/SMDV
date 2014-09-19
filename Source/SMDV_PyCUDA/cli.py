# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 13:47:34 2014

@author: HP
"""

import click
import scipy.io
#import os
from numpy import average as avr
 
@click.command()
#@click.option('--count', default=1, help='number of greetings')/
@click.option('-b', '--block', default=128, help='Block size for CUDA. Default: 128')
@click.option('-ss', '--slice-size', 'ss', default=64, help='Slice size for ...Ellpack. Default: 64')
@click.option('-tpr', '--thread-per-row', 'tpr', default=2, help='Thread per row. Default: 2')
@click.option('-a', '--align', default=32, help='Align const for Ellpack. Default: 32')
@click.option('-p', '--prefetch', default=2, help='PreFetch for SlicedEllpack. Default: 2')
@click.option('-r', '--repeat', default=1, help='Count of repetitions calculations. Deafult: 1')
@click.option('-cf', '--confidence-factor', default=0.0005, help='Confidence interval for test multiplication. Default: 0.0005')

@click.option('-ell', '--ellpack', 'ell', is_flag=True, help='Use Ellpack format')
@click.option('-sle', '--sliced', 'sle', is_flag=True, help='Use Sliced Ellpack format')
@click.option('-see', '--sertilp', 'see', is_flag=True, help='Use Sertilp Ellpack format')
@click.option('-ert', '--ertilp', 'ert', is_flag=True, help='Use Ertilp format')
@click.option('-cpu', '--cpu', 'cpu', is_flag=True, help='Use CPU method multiplication (without GPU) on Numpy')

@click.option('-pm', '--printMatrix', 'pm', is_flag=True, help='Print matrix representation')
@click.option('-c', '--convertion', 'conv', is_flag=True, help='Print converted matrix to format')
@click.option('-m', '--multiply', is_flag=True, help='Multiply matrix to vector (1, 2, 3, ...)')

@click.option('-rst', '--result', is_flag=True, help='Print result multiplication')
@click.option('-t', '--time', is_flag=True, help='Print list of time multiplication')
@click.option('-avr', '--avrtime', is_flag=True, help='Print average time multiplication')
@click.option('--test', is_flag=True, help='Testing result multiplication. Print bad row')

@click.option('-q', '--quite', is_flag=True, help='Without messages, only effects functions.')
@click.option('--lang', default='en', type=click.Choice(['en', 'pl']), help='Language messages.')

#@click.argument('folder', nargs=1, type=click.Path(exists=True))
#@click.argument('matrices', nargs=-1, required=True)
@click.argument('matrices', nargs=-1, required=True, type=click.Path(exists=True))

def cli(block, ss, tpr, align, prefetch, repeat, confidence_factor, ell, sle, see, ert, cpu, pm, conv, multiply, result, time, avrtime, test, quite, lang, matrices):
    colors = {
        'success' : 'green',
        'info' : 'cyan',
        'warning' : 'red',
        'danger' : 'yellow'
    }    
    for matrixFilename in matrices:
#        matrixPath = str(os.path.join(folder, matrixFilename))
        matrixPath = str(matrixFilename)
        matrix = scipy.io.mmread(matrixPath)
        if not quite: click.secho(getMessage('title', lang) + matrixFilename, fg=colors['success']) 
        if pm:
            if not quite: click.secho(getMessage('pm', lang), fg=colors['info'])
            printMatrix(matrix)
        if conv:
            from matrixUtilites import stringListInList
            if not quite: click.secho(getMessage('conv', lang), fg=colors['info'])
            if ell:
                if not quite: click.secho(getMessage('convEll', lang), fg=colors['danger'])
                from matrixFormat import convertToELL
                click.echo(stringListInList(convertToELL(matrix, array=False)))
            if sle:
                if not quite: click.secho(getMessage('convSliced', lang), fg=colors['danger'])
                from matrixFormat import convertToSlicedELL
                click.echo(stringListInList(convertToSlicedELL(matrix, array=False, watkiNaWiersz=tpr, sliceSize=ss, align=align)))
            if see:
                if not quite: click.secho(getMessage('convSertilp', lang), fg=colors['danger'])
                from matrixFormat import convertToSertilpELL
                click.echo(stringListInList(convertToSertilpELL(matrix, array=False, watkiNaWiersz=tpr, sliceSize=ss, align=align, prefetch=prefetch)))
            if ert:
                if not quite: click.secho(getMessage('convErtilp', lang), fg=colors['danger'])
                from matrixFormat import convertToErtilp
                click.echo(stringListInList(convertToErtilp(matrix, threadPerRow=tpr, prefetch=prefetch, array=False)))
        if multiply:
            if not quite: click.secho(getMessage('multiply', lang), fg=colors['info'])
            resultNumpy = ''
            if cpu:
                if not quite: click.secho(getMessage('multiplyCpu', lang), fg=colors['danger'])
                from matrixMultiplication import multiplyCPU
                resultMultiply = multiplyCPU(matrix, repeat=repeat)
                if test: resultNumpy = resultMultiply[0]
                resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, quite=quite, lang=lang)
            elif test:
                from matrixMultiplication import multiplyCPU
                resultNumpy = multiplyCPU(matrix, repeat=repeat)[0]
            if ell:
                if not quite: click.secho(getMessage('multiplyEll', lang), fg=colors['danger'])
                from matrixMultiplication import multiplyELL
                resultMultiply = multiplyELL(matrix, repeat=repeat, blockSize=block)
                resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, quite=quite, lang=lang)
                if test: testResult(resultNumpy, resultMultiply[0], confidence_factor, quite, lang)
            if sle:
                if not quite: click.secho(getMessage('multiplySliced', lang), fg=colors['danger'])
                from matrixMultiplication import multiplySlicedELL
                resultMultiply = multiplySlicedELL(matrix, alignConst=align, sliceSize=ss, threadPerRow=tpr, repeat=repeat)
                resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, quite=quite, lang=lang)
                if test: testResult(resultNumpy, resultMultiply[0], confidence_factor, quite, lang)
            if see:
                if not quite: click.secho(getMessage('multiplySertilp', lang), fg=colors['danger'])
                from matrixMultiplication import multiplySertilp
                resultMultiply = multiplySertilp(matrix, alignConst=align, sliceSize=ss, threadPerRow=tpr, prefetch=prefetch, repeat=repeat)
                resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, quite=quite, lang=lang)
                if test: testResult(resultNumpy, resultMultiply[0], confidence_factor, quite, lang)
            if ert:
                if not quite: click.secho(getMessage('multiplyErtilp', lang), fg=colors['danger'])
                from matrixMultiplication import multiplyErtilp
                resultMultiply = multiplyErtilp(matrix, blockSize=block, threadPerRow=tpr, prefetch=prefetch, repeat=repeat)
                resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, quite=quite, lang=lang)
                if test: testResult(resultNumpy, resultMultiply[0], confidence_factor, quite, lang)
                    
def resumeResult(resultMuliply, resultPrint, timePrint, avrTimePrint, quite, lang):
    if resultPrint:
        click.echo(('' if quite else getMessage('result', lang)) + str(resultMuliply[0]))
    if timePrint:
        click.echo(('' if quite else getMessage('timeList', lang)) + str(resultMuliply[1]))
    if avrTimePrint:
        click.echo(('' if quite else getMessage('avrTime', lang)) + str(avr(resultMuliply[1])))
def testResult(model, check, confidenceFactor, quite, lang):
    click.echo(('' if quite else getMessage('test', lang)) + str(resultEquals(model, check, confidenceFactor)))
def getMessage(idMessage, lang='en'):
    if lang == 'pl':
        return {
            'error' : 'error',
            'title' : 'Przetwarzanie macierzy: ',
            'pm' : 'Reprezentacja danych w macierzy: ',
            'conv' : u'Reprezentacje macierzy po konwersji do wybranych formatów: ',
            'convEll' : 'Konwersja do ELLPACK',
            'convSliced' : 'Konwersja do SLICED ELLPACK',
            'convSertilp' : 'Konwersja do SERTILP',
            'convErtilp' : 'Konwersja do ERTILP',
            'multiply' : u'Mnożenie macierzy w poszczególnych formatach: ',
            'multiplyCpu' : u'Mnożenie przy użyciu Numpy (tylko CPU)',
            'multiplyEll' : u'Mnożenie formatem ELLPACK',
            'multiplySertilp' : u'Mnożenie formatem SERTILP',
            'multiplySliced' : u'Mnożenie formatem SLICED',
            'multiplyErtilp' : u'Mnożenie formatem ERTILP',
            'result' : u'Wynik: ',
            'timeList' : u'Lista czasów [ms]: ',
            'avrTime' : u'Średni czas [ms]: ',
            'test': u'Błędy (pozycja, różnica): '
        }.get(idMessage, 'error')
    elif lang == 'en':
        return {
            'error' : 'error',
            'title' : 'Process matrix: ',
            'pm' : 'Representation of data matrix: ',
            'conv' : u'Representations of the matrix after conversion to selected formats: ',
            'convEll' : 'Conversion to ELLPACK',
            'convSliced' : 'Conversion to SLICED ELLPACK',
            'convSertilp' : 'Conversion to SERTILP',
            'convErtilp' : 'Conversion to ERTILP',
            'multiply' : u'Matrix multiplication in the various formats: ',
            'multiplyCpu' : u'Multiplication with Numpy (only CPU)',
            'multiplyEll' : u'Multiplication with ELLPACK',
            'multiplySertilp' : u'Multiplication with SERTILP',
            'multiplySliced' : u'Multiplication with SLICED',
            'multiplyErtilp' : u'Multiplication with ERTILP',
            'result' : u'Result: ',
            'timeList' : u'List of times multiplication [ms]: ',
            'avrTime' : u'Average time [ms]: ',
            'test': 'Errors (position, different): '
        }.get(idMessage, 'error')
    else:
        return 'Not implement language: ' + lang

def printMatrix(matrixFile):
    click.echo(str(matrixFile))
    
def resultEquals(correct, current, confidenceFactor = 0.05):
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
            if current[i] > correct[i]*(1+confidenceFactor) or current[i] < correct[i]*(1-confidenceFactor):
                result.append((i, correct[i] - current[i]))
    for i in range(endMin, len(objMax)):
        if objMax[i] != 0:
            return result.append((i, '#'))
    return result
          
if __name__ == '__main__':
    cli()
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 13:47:34 2014

@author: HP
"""

import click
import scipy.io
import os
from numpy import average as avr
 
@click.command()
#@click.option('--count', default=1, help='number of greetings')/
@click.option('--block', default=32, help='Block size for CUDA')
@click.option('--ss', default=64, help='Slice size for ...Ellpack')
@click.option('--tpr', default=2, help='Thread per row')
@click.option('--align', default=32, help='Align const for Ellpack')
@click.option('--prefetch', default=2, help='PreFetch for SlicedEllpack')
@click.option('--repeat', default=1, help='Count of repetitions calculations')
@click.option('--pt', default=3, help='Precision of time')

@click.option('--ell/--no-ell', default=False, help='Use Ellpack format')
@click.option('--sle/--no-sle', default=False, help='Use Sliced Ellpack format')
@click.option('--see/--no-see', default=False, help='Use Sertilp Ellpack format')
@click.option('--ert/--no-ert', default=False, help='Use Ertilp format')
@click.option('--cpu/--no-cpu', default=False, help='Use CPU method multiplication (without GPU) on Numpy')

@click.option('--pm/--no-pm', default=False, help='Print matrix representation')
@click.option('--conv/--no-conv', default=False, help='Print converted matrix to format')
@click.option('--multiply/--no-multiply', default=False, help='Multiply matrix to vector (1, 2, 3, ...)')

@click.option('--result/--no-result', default=False, help='Print result multiplication')
@click.option('--time/--no-time', default=False, help='Print list of time multiplication')
@click.option('--avrtime/--no-avrtime', default=True, help='Print average time multiplication')

@click.option('--quite/--no-quite', default=False, help='Without messages, only effects functions.')
@click.option('--lang', default='en', help='Language messages. Polish (pl) or english (en).')

@click.argument('folder', nargs=1, type=click.Path(exists=True))
@click.argument('matrices', nargs=-1, required=True)

def cli(block, ss, tpr, align, prefetch, repeat, pt, ell, sle, see, ert, cpu, pm, conv, multiply, result, time, avrtime, quite, lang, folder, matrices):
    colors = {
        'success' : 'green',
        'info' : 'blue',
        'warning' : 'yelow',
        'danger' : 'red'
    }    
    for matrixFilename in matrices:
        matrixPath = str(os.path.join(folder, matrixFilename))
        matrix = scipy.io.mmread(matrixPath)
        if not quite: click.echo(getMessage('title', lang) + matrixFilename, color=colors['success']) 
        if pm:
            if not quite: click.echo(getMessage('pm', lang), color=colors['info'])
            printMatrix(matrix)
        if conv:
            from matrixUtilites import stringListInList
            if not quite: click.echo(getMessage('conv', lang), color=colors['info'])
            if ell:
                if not quite: click.echo(getMessage('convEll', lang), color=colors['danger'])
                from matrixFormat import convertToELL
                click.echo(stringListInList(convertToELL(matrix, array=False)))
            if sle:
                if not quite: click.echo(getMessage('convSliced', lang), color=colors['danger'])
                from matrixFormat import convertToSlicedELL
                click.echo(stringListInList(convertToSlicedELL(matrix, array=False, watkiNaWiersz=tpr, sliceSize=ss, align=align)))
            if see:
                if not quite: click.echo(getMessage('convSertilp', lang), color=colors['danger'])
                from matrixFormat import convertToSertilpELL
                click.echo(stringListInList(convertToSertilpELL(matrix, array=False, watkiNaWiersz=tpr, sliceSize=ss, align=align, prefetch=prefetch)))
            if ert:
                if not quite: click.echo(getMessage('convErtilp', lang), color=colors['danger'])
                from matrixFormat import convertToErtilp
                click.echo(stringListInList(convertToErtilp(matrix, threadPerRow=tpr, prefetch=prefetch, array=False)))
        if multiply:
            if not quite: click.echo(getMessage('multiply', lang), color=colors['info'])
            if cpu:
                if not quite: click.echo(getMessage('multiplyCpu', lang), color=colors['danger'])
                from matrixMultiplication import multiplyCPU
                resultMultiply = multiplyCPU(matrix, repeat=repeat)
                resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, quite=quite, lang=lang)
            if ell:
                if not quite: click.echo(getMessage('multiplyEll', lang), color=colors['danger'])
                from matrixMultiplication import multiplyELL
                resultMultiply = multiplyELL(matrix, repeat=repeat, blockSize=block)
                resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, quite=quite, lang=lang)
            if sle:
                if not quite: click.echo(getMessage('multiplySliced', lang), color=colors['danger'])
                from matrixMultiplication import multiplySlicedELL
                resultMultiply = multiplySlicedELL(matrix, alignConst=align, sliceSize=ss, threadPerRow=tpr, repeat=repeat)
                resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, quite=quite, lang=lang)
            if see:
                if not quite: click.echo(getMessage('multiplySertilp', lang), color=colors['danger'])
                from matrixMultiplication import multiplySertilp
                resultMultiply = multiplySertilp(matrix, alignConst=align, sliceSize=ss, threadPerRow=tpr, prefetch=prefetch, repeat=repeat)
                resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, quite=quite, lang=lang)
            if ert:
                if not quite: click.echo(getMessage('multiplyErtilp', lang), color=colors['danger'])
                from matrixMultiplication import multiplyErtilp
                resultMultiply = multiplyErtilp(matrix, alignConst=align, blockSize=block, threadPerRow=tpr, prefetch=prefetch, repeat=repeat)
                resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, quite=quite, lang=lang)
            
def resumeResult(resultMuliply, resultPrint, timePrint, avrTimePrint, quite, lang):
    if resultPrint:
        click.echo(('' if quite else getMessage('result', lang)) + str(resultMuliply[0]))
    if timePrint:
        click.echo(('' if quite else getMessage('timeList', lang)) + str(resultMuliply[1]))
    if avrTimePrint:
        click.echo(('' if quite else getMessage('avrTime', lang)) + str(avr(resultMuliply[1])))
def getMessage(idMessage, lang='pl'):
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
            'avrTime' : u'Średni czas [ms]: '
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
            'avrTime' : u'Average time [ms]: '
        }.get(idMessage, 'error')
    else:
        return 'Not implement language: ' + lang

def printMatrix(matrixFile):
    click.echo(str(matrixFile))
          
if __name__ == '__main__':
    cli()
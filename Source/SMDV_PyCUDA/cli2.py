# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 13:47:34 2014

@author: HP
"""

import click
import scipy.io
from os.path import basename, isfile
from numpy import average as avr, std as nstd

colors = {
        'success' : 'green',
        'info' : 'cyan',
        'warning' : 'red',
        'danger' : 'yellow'
    }
 
@click.group(chain=True)
@click.option('-q', '--quite', is_flag=True, help='Without messages, only effects functions.')
@click.option('--lang', default='en', type=click.Choice(['en', 'pl']), help='Language messages.')
@click.option('-sep', '--separator', 'sep',  default='; ', help='Separator for data in report. Default: "; "')
@click.option('-eol', '--end-of-line', 'eol', default='\r\n', help=r'End of line for data in report. Default Windows style: "\r\n".')

@click.argument('matrix-path', nargs=1, required=True, type=click.Path(exists=True))
@click.pass_context
def cli(ctx, quite, lang, sep, eol, matrix_path):
    matrix = scipy.io.mmread(str(matrix_path))
    ctx.obj['quite'] = quite
    ctx.obj['lang'] = lang
    ctx.obj['matrix'] = matrix
    ctx.obj['filename'] = basename(matrix_path)
    ctx.obj['sep'] = sep
    ctx.obj['eol'] = eol
    if not quite: click.secho(getMessage('title', lang) + matrix_path, fg=colors['success'])

@cli.command()
@click.option('-o', '--output', type=click.File(mode='a', lazy=True), help='File to save raport. Format CSV. If exist append new data.')
@click.pass_context
def info(ctx, output):
    '''
    Print info about matrix - rows, cols, nnz and sparsing.
    '''
    matrix = ctx.obj['matrix']
    quite = ctx.obj['quite']
    lang = ctx.obj['lang']
    if not quite: click.secho(getMessage('info_title', lang), fg=colors['info'])
    shape = matrix.shape
    nnz = matrix.nnz
    sparsing = round( ( (nnz+0.0)/ ( (shape[0] * shape[1]) ) ) * 100, 2)
    nnz = str(nnz)
    sparsing = str(sparsing)
    click.echo(('' if quite else getMessage('info_rows', lang)) + str(shape[0]))
    click.echo(('' if quite else getMessage('info_cols', lang)) + str(shape[1]))
    click.echo(('' if quite else getMessage('info_nnz', lang)) + nnz)
    click.echo(('' if quite else getMessage('info_sparse', lang)) + sparsing + '%')
    if output:
        sep = ctx.obj['sep']
        eol = ctx.obj['eol']
        if not isfile(output.name):
            headers = ['matrix', 'rows', 'cols', 'nnz', 'sparsing']
            output.write(sep.join(headers) + eol)
        data = [ctx.obj['filename'], str(shape[0]), str(shape[1]), nnz, sparsing]
        output.write(sep.join(data) + eol )
        
@cli.command()
@click.pass_context
def pm(ctx):
    '''
    Print matrix
    '''
    matrix = ctx.obj['matrix']
    quite = ctx.obj['quite']
    lang = ctx.obj['lang']

    if not quite: click.secho(getMessage('pm', lang), fg=colors['info'])
    printMatrix(matrix)
        
def printMatrix(matrixFile):
    click.echo(str(matrixFile))
    
@cli.command()
@click.option('-b', '--block', default=128, help='Block size for CUDA. Default: 128')
@click.option('-ss', '--slice-size', 'ss', default=64, help='Slice size for ...Ellpack. Default: 64')
@click.option('-tpr', '--thread-per-row', 'tpr', default=2, help='Thread per row. Default: 2')
@click.option('-a', '--align', default=32, help='Align const for Ellpack. Default: 32')
@click.option('-p', '--prefetch', default=2, help='PreFetch for SlicedEllpack. Default: 2')

@click.option('-ell', '--ellpack', 'ell', is_flag=True, help='Use Ellpack format')
@click.option('-sle', '--sliced', 'sle', is_flag=True, help='Use Sliced Ellpack format')
@click.option('-see', '--sertilp', 'see', is_flag=True, help='Use Sertilp Ellpack format')
@click.option('-ert', '--ertilp', 'ert', is_flag=True, help='Use Ertilp format')

@click.pass_context
def conv(ctx, block, ss, tpr, align, prefetch, ell, sle, see, ert):
    '''
    Convert matrix to formats matrix.
    '''
    matrix = ctx.obj['matrix']
    quite = ctx.obj['quite']
    lang = ctx.obj['lang']
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

@cli.command()
@click.option('-b', '--block', default=128, help='Block size for CUDA. Default: 128')
@click.option('-ss', '--slice-size', 'ss', default=64, help='Slice size for ...Ellpack. Default: 64')
@click.option('-tpr', '--thread-per-row', 'tpr', default=2, help='Thread per row. Default: 2')
@click.option('-a', '--align', default=32, help='Align const for Ellpack. Default: 32')
@click.option('-p', '--prefetch', default=2, help='PreFetch for SlicedEllpack. Default: 2')
@click.option('-r', '--repeat', default=1, help='Count of repetitions calculations. Deafult: 1')

@click.option('-ell', '--ellpack', 'ell', is_flag=True, help='Use Ellpack format')
@click.option('-sle', '--sliced', 'sle', is_flag=True, help='Use Sliced Ellpack format')
@click.option('-see', '--sertilp', 'see', is_flag=True, help='Use Sertilp Ellpack format')
@click.option('-ert', '--ertilp', 'ert', is_flag=True, help='Use Ertilp format')
@click.option('-cpu', '--cpu', 'cpu', is_flag=True, help='Use CPU method multiplication (without GPU) on Numpy')

@click.option('-rst', '--result', is_flag=True, help='Print result multiplication')
@click.option('-t', '--time', is_flag=True, help='Print list of time multiplication')
@click.option('-avr', '--avrtime', is_flag=True, help='Print average time multiplication')
@click.option('-std', '--standard-deviation', 'std', is_flag=True, help='Print standard deviation of time multiplication')
@click.option('--test', flag_value=0.0005, help='Testing result multiplication. Print bad row. Value is confidence factor.')

@click.option('-o', '--output', type=click.File(mode='a', lazy=True), help='File to save raport. Format CSV. If exist append new data.')

@click.pass_context
def multiply(ctx, block, ss, tpr, align, prefetch, ell, sle, see, ert, cpu, repeat, result, time, avrtime, std, test, output):
    '''Multiplication matrix in formats...'''    
    matrix = ctx.obj['matrix']
    quite = ctx.obj['quite']
    lang = ctx.obj['lang']
    sep = ctx.obj['sep']
    eol = ctx.obj['eol']
    if not quite: click.secho(getMessage('multiply', lang), fg=colors['info'])
    if output and not isfile(output.name):
        headers = ['matrix', 'format', 'average time', 'standard deviation time', 'times']
        output.write(sep.join(headers) + eol)
    resultNumpy = ''
    if cpu:
        if not quite: click.secho(getMessage('multiplyCpu', lang), fg=colors['danger'])
        from matrixMultiplication import multiplyCPU
        resultMultiply = multiplyCPU(matrix, repeat=repeat)
        if test: resultNumpy = resultMultiply[0]
        resumeResult(ctx=ctx, resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, lang=lang, output=output, formatName='cpu')
    elif test:
        from matrixMultiplication import multiplyCPU
        resultNumpy = multiplyCPU(matrix, repeat=repeat)[0]
    if ell:
        if not quite: click.secho(getMessage('multiplyEll', lang), fg=colors['danger'])
        from matrixMultiplication import multiplyELL
        resultMultiply = multiplyELL(matrix, repeat=repeat, blockSize=block)
        resumeResult(ctx=ctx, resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, lang=lang, output=output, formatName='ellpack')
        if test: testResult(resultNumpy, resultMultiply[0], test, quite, lang)
    if sle:
        if not quite: click.secho(getMessage('multiplySliced', lang), fg=colors['danger'])
        from matrixMultiplication import multiplySlicedELL
        resultMultiply = multiplySlicedELL(matrix, alignConst=align, sliceSize=ss, threadPerRow=tpr, repeat=repeat)
        resumeResult(ctx=ctx, resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, lang=lang, output=output, formatName='sliced')
        if test: testResult(resultNumpy, resultMultiply[0], test, quite, lang)
    if see:
        if not quite: click.secho(getMessage('multiplySertilp', lang), fg=colors['danger'])
        from matrixMultiplication import multiplySertilp
        resultMultiply = multiplySertilp(matrix, alignConst=align, sliceSize=ss, threadPerRow=tpr, prefetch=prefetch, repeat=repeat)
        resumeResult(ctx=ctx, resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, lang=lang, output=output, formatName='sertilp')
        if test: testResult(resultNumpy, resultMultiply[0], test, quite, lang)
    if ert:
        if not quite: click.secho(getMessage('multiplyErtilp', lang), fg=colors['danger'])
        from matrixMultiplication import multiplyErtilp
        resultMultiply = multiplyErtilp(matrix, blockSize=block, threadPerRow=tpr, prefetch=prefetch, repeat=repeat)
        resumeResult(ctx=ctx, resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, lang=lang, output=output, formatName='ertilp')
        if test: testResult(resultNumpy, resultMultiply[0], test, quite, lang)
         
def resumeResult(ctx, resultMuliply, resultPrint, timePrint, avrTimePrint, stdTimePrint, quite, lang, output, formatName):
    if resultPrint:
        click.echo(('' if quite else getMessage('result', lang)) + str(resultMuliply[0]))
    if timePrint:
        click.echo(('' if quite else getMessage('timeList', lang)) + str(resultMuliply[1]))
    if avrTimePrint:
        click.echo(('' if quite else getMessage('avrTime', lang)) + str(avr(resultMuliply[1])))
    if stdTimePrint:
        click.echo(('' if quite else getMessage('stdTime', lang)) + str(nstd(resultMuliply[1])))
    if output:
        filename = ctx.obj['filename']
        sep = ctx.obj['sep']
        eol = ctx.obj['eol']
        avrTime = str(avr(resultMuliply[1]))
        stdTime = str(nstd(resultMuliply[1]))
        data = [filename, formatName, avrTime, stdTime].extend(map(str, resultMuliply[1]))
        output.write(sep.join(data) + eol )
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
            'test': 'Errors (position, different): ',
            'info_rows': 'Rows: ',
            'info_cols': 'Cols: ',
            'info_nnz': 'NNZ: ',
            'info_sparse': 'Sparsing: ',
            'info_title': 'Info about matrix: '
        }.get(idMessage, 'error')
    else:
        return 'Not implement language: ' + lang
  
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
    cli(obj={})
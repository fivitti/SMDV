# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 13:47:34 2014

@author: Sławomir Figiel
"""

import click
import scipy.io
from os.path import basename, isfile
from numpy import average as avr, std as nstd

colors = {
        'success' : 'green',
        'info' : 'cyan',
        'danger' : 'red',
        'warning' : 'yellow'
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
@click.option('-v', '--vector-path', 'vec', type=click.Path(exists=True))
@click.pass_context
def pm(ctx, vec):
    '''
    Print matrix
    '''
    matrix = ctx.obj['matrix']
    quite = ctx.obj['quite']
    lang = ctx.obj['lang']

    if not quite: click.secho(getMessage('pm', lang), fg=colors['info'])
    printMatrix(matrix)
    if vec:
        if not quite: click.secho(getMessage('vec', lang), fg=colors['info'])
        from numpy import load
        from matrixUtilites import stringVector
        click.echo(stringVector(load(str(vec))))
        
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
        if not quite: click.secho(getMessage('convEll', lang), fg=colors['warning'])
        from matrixFormat import convertToELL
        click.echo(stringListInList(convertToELL(matrix, array=False)))
    if sle:
        if not quite: click.secho(getMessage('convSliced', lang), fg=colors['warning'])
        from matrixFormat import convertToSlicedELL
        click.echo(stringListInList(convertToSlicedELL(matrix, array=False, watkiNaWiersz=tpr, sliceSize=ss, align=align)))
    if see:
        if not quite: click.secho(getMessage('convSertilp', lang), fg=colors['warning'])
        from matrixFormat import convertToSertilpELL
        click.echo(stringListInList(convertToSertilpELL(matrix, array=False, watkiNaWiersz=tpr, sliceSize=ss, align=align, prefetch=prefetch)))
    if ert:
        if not quite: click.secho(getMessage('convErtilp', lang), fg=colors['warning'])
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

@click.option('-rst', '--result', is_flag=True, help='Print result multiplication.')
@click.option('-t', '--time', is_flag=True, help='Print list of time multiplication')
@click.option('-avr', '--avrtime', is_flag=True, help='Print average time multiplication')
@click.option('-std', '--standard-deviation', 'std', is_flag=True, help='Print standard deviation of time multiplication')
@click.option('--test', type=click.FLOAT, help='Testing result multiplication. Print bad row. Value is confidence factor.')
@click.option('-com', '--compensate', 'com', type=click.INT, help='N first time are remove (returned times decremented by n). Recommended in testing the speed, because the n first times (e. g. one) are a long delay.' )
@click.option('-o', '--output', type=click.File(mode='a', lazy=True), help='File to save raport. Format CSV. If exist append new data.')

@click.argument('vector-path', nargs=1, required=True, type=click.Path(exists=True))

@click.pass_context
def multiply(ctx, block, ss, tpr, align, prefetch, ell, sle, see, ert, cpu, repeat, result, time, avrtime, std, test, com, output, vector_path):
    '''Multiplication matrix in formats...'''   
    if com: 
        repeat += com
    matrix = ctx.obj['matrix']
    quite = ctx.obj['quite']
    lang = ctx.obj['lang']
    sep = ctx.obj['sep']
    eol = ctx.obj['eol']
    
    from numpy import load
    vector = load(str(vector_path))
    if not len(vector) == matrix.shape[1]:
        raise click.BadParameter('Length of the vector is not equal to the number of columns of the matrix.')
        
    if not quite: click.secho(getMessage('multiply', lang), fg=colors['info'])
    if output and not isfile(output.name):
        headers = ['matrix', 'format', 'average time', 'standard deviation time', 'times']
        output.write(sep.join(headers) + eol)
    resultNumpy = ''
    if cpu:
        if not quite: click.secho(getMessage('multiplyCpu', lang), fg=colors['warning'])
        from matrixMultiplication import multiplyCPU
        resultMultiply = multiplyCPU(matrix, repeat=repeat, vector=vector)
        if test: resultNumpy = resultMultiply[0]
        resumeResult(ctx=ctx, resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, lang=lang, output=output, formatName='cpu', compensate=com)
    elif test:
        from matrixMultiplication import multiplyCPU
        resultNumpy = multiplyCPU(matrix, repeat=repeat)[0]
    if ell:
        if not quite: click.secho(getMessage('multiplyEll', lang), fg=colors['warning'])
        from matrixMultiplication import multiplyELL
        resultMultiply = multiplyELL(matrix, vector=vector, repeat=repeat, blockSize=block)
        resumeResult(ctx=ctx, resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, lang=lang, output=output, formatName='ellpack', compensate=com)
        if test: testResult(resultNumpy, resultMultiply[0], test, quite, lang)
    if sle:
        if not quite: click.secho(getMessage('multiplySliced', lang), fg=colors['warning'])
        from matrixMultiplication import multiplySlicedELL
        resultMultiply = multiplySlicedELL(matrix, vector=vector, alignConst=align, sliceSize=ss, threadPerRow=tpr, repeat=repeat)
        resumeResult(ctx=ctx, resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, lang=lang, output=output, formatName='sliced', compensate=com)
        if test: testResult(resultNumpy, resultMultiply[0], test, quite, lang)
    if see:
        if not quite: click.secho(getMessage('multiplySertilp', lang), fg=colors['warning'])
        from matrixMultiplication import multiplySertilp
        resultMultiply = multiplySertilp(matrix, vector=vector, alignConst=align, sliceSize=ss, threadPerRow=tpr, prefetch=prefetch, repeat=repeat)
        resumeResult(ctx=ctx, resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, lang=lang, output=output, formatName='sertilp', compensate=com)
        if test: testResult(resultNumpy, resultMultiply[0], test, quite, lang)
    if ert:
        if not quite: click.secho(getMessage('multiplyErtilp', lang), fg=colors['warning'])
        from matrixMultiplication import multiplyErtilp
        resultMultiply = multiplyErtilp(matrix, vector=vector, blockSize=block, threadPerRow=tpr, prefetch=prefetch, repeat=repeat)
        resumeResult(ctx=ctx, resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, lang=lang, output=output, formatName='ertilp', compensate=com)
        if test: testResult(resultNumpy, resultMultiply[0], test, quite, lang)
         
def resumeResult(ctx, resultMuliply, resultPrint, timePrint, avrTimePrint, stdTimePrint, quite, lang, output, formatName, compensate):
    times = resultMuliply[1]
    if compensate:
        times = times[compensate:]
    if resultPrint:
        from matrixUtilites import stringVector
        click.echo(('' if quite else getMessage('result', lang)) + stringVector(resultMuliply[0]))
    if timePrint:
        click.echo(('' if quite else getMessage('timeList', lang)) + str(times))
    if avrTimePrint:
        click.echo(('' if quite else getMessage('avrTime', lang)) + str(avr(times)))
    if stdTimePrint:
        click.echo(('' if quite else getMessage('stdTime', lang)) + str(nstd(times)))
    if output:
        filename = ctx.obj['filename']
        sep = ctx.obj['sep']
        eol = ctx.obj['eol']
        avrTime = str(avr(times))
        stdTime = str(nstd(times))
        data = [filename, formatName, avrTime, stdTime]
        data.extend(map(str, times))
        output.write(sep.join(data) + eol )
def testResult(model, check, confidenceFactor, quite, lang):
    from matrixUtilites import resultEquals
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
            'info_title': 'Info about matrix: ',
            'vec': 'Representation of data vector: '
        }.get(idMessage, 'error')
    else:
        return 'Not implement language: ' + lang
          
if __name__ == '__main__':
    cli(obj={})
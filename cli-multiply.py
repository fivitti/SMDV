# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 13:47:34 2014

@author: Sławomir Figiel
"""

import click
import scipy.io
from os.path import isfile
from numpy import average as avr, std as nstd, load
from matrixMultiplication import multiply_cpu, multiply_ellpack, multiply_sliced, multiply_sertilp, multiply_ertilp, multiply_csr
from matrixUtilites import stringVector, resultEquals, dictVectorPaths
from filesUtilites import pathReduction, sortPaths

colors = {
        'success' : 'green',
        'info' : 'cyan',
        'danger' : 'red',
        'warning' : 'yellow'
    }
 
@click.command()
@click.option('-b', '--block', default=128, help='Block size for CUDA. Default: 128')
@click.option('-ss', '--slice-size', 'ss', default=64, help='Slice size for ...Ellpack. Default: 64')
@click.option('-tpr', '--thread-per-row', 'tpr', default=2, help='Thread per row. Default: 2')
@click.option('-a', '--align', default=32, help='Align const for Ellpack. Default: 32')
@click.option('-p', '--prefetch', default=2, help='PreFetch for SlicedEllpack. Default: 2')
@click.option('-r', '--repeat', default=1, help='Count of repetitions calculations. Deafult: 1')

@click.option('-csr', '--csr', 'csr', is_flag=True, help='Use CSR format')
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
@click.option('-o', '--output', type=click.File(mode='a', lazy=True), help='File to save raport. Format CSV. If exist append new data. Added to the file it info if it is created.')
@click.option('-param', '--parameters', is_flag=True, help='Print value of parameters.')

@click.option('-q', '--quite', is_flag=True, help='Without messages, only effects functions.')
@click.option('-sep', '--separator', 'sep',  default='; ', help='Separator for data in report. Default: "; "')
#@click.option('-eol', '--end-of-line', 'eol_ch', default='lin', type=click.Choice(['win', 'lin']), help=r'End of line for data in report. Default Linux style: "\n".')

@click.argument('vector-path', nargs=1, required=True, type=click.Path(exists=True))
@click.argument('matrix-path', nargs=1, required=True, type=click.Path(exists=True))
def cli(block, ss, tpr, align, prefetch, csr, ell, sle, see, ert, cpu, repeat, result, time, avrtime, std, test, com, output, parameters, vector_path, quite, sep, matrix_path):
    eol = '\n'
    param = {
            'Block' : str(block),
            'Slice size' : str(ss),
            'Threads per row' : str(tpr),
            'Align' : str(align),
            'Prefetch' : str(prefetch),
            'Repeat' : str(repeat),
            'Compensate' : str(com) if com else '0'
        }
    vectorsDict = dictVectorPaths(sortPaths(pathReduction([str(vector_path),]), '.npy')['.npy'])
    matricesPaths = sortPaths(pathReduction([str(matrix_path),]), '.mtx')['.mtx']
    
    if com: 
        repeat += com

    if parameters:
        if not quite: click.secho(getMessage('paramInfo'), fg=colors['info'])
        paramRows = []
        for k, v in param.items():
            paramRows.append('  {0:<20}{1}'.format(k, v))
        click.echo('\n'.join(paramRows))
    
    if output and not isfile(output.name):
        output.write(sep.join(param.keys()) + eol)
        output.write(sep.join(param.values()) + eol)
        output.write(eol)
        headers = ['matrix', 'format', 'average time', 'standard deviation time', 'times']
        output.write(sep.join(headers) + eol)
    
    for matrixPath in matricesPaths:
        try:
            matrix = scipy.io.mmread(matrixPath)
        except:
            click.secho(getMessage('open_failed') % matrixPath, fg=colors['danger'])
            continue
        cols = matrix.shape[1]
        if not cols in vectorsDict.keys():
            click.secho(getMessage('bad_length') % matrixPath, fg=colors['danger'])
            continue
        vectorPath = vectorsDict[cols]
        if not quite: click.secho(getMessage('multiply') % (matrixPath, vectorPath), fg=colors['success'])
        vector = load(vectorPath)
        resultNumpy = ''
        if cpu:
            if not quite: click.secho(getMessage('multiplyCpu'), fg=colors['warning'])
            resultMultiply = multiply_cpu(matrix, repeat=repeat, vector=vector)
            if test: resultNumpy = resultMultiply[0]
            resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, output=output, formatName='cpu', compensate=com, matrixName=matrixPath, sep=sep, eol=eol)
        elif test:
            resultNumpy = multiply_cpu(matrix, vector=vector, repeat=repeat)[0]
        if csr:
            if not quite: click.secho(getMessage('multiply_csr'), fg=colors['warning'])
            resultMultiply = multiply_csr(matrix, vector=vector, repeat=repeat, block_size=block)
            resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, output=output, formatName='csr', compensate=com, matrixName=matrixPath, sep=sep, eol=eol)
            if test: testResult(resultNumpy, resultMultiply[0], test, quite)            
        if ell:
            if not quite: click.secho(getMessage('multiplyEll'), fg=colors['warning'])
            resultMultiply = multiply_ellpack(matrix, vector=vector, repeat=repeat, block_size=block)
            resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, output=output, formatName='ellpack', compensate=com, matrixName=matrixPath, sep=sep, eol=eol)
            if test: testResult(resultNumpy, resultMultiply[0], test, quite)
        if sle:
            if not quite: click.secho(getMessage('multiplySliced'), fg=colors['warning'])
            resultMultiply = multiply_sliced(matrix, vector=vector, align=align, slice_size=ss, threads_per_row=tpr, repeat=repeat)
            resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, output=output, formatName='sliced', compensate=com, matrixName=matrixPath, sep=sep, eol=eol)
            if test: testResult(resultNumpy, resultMultiply[0], test, quite)
        if see:
            if not quite: click.secho(getMessage('multiply_sertilp'), fg=colors['warning'])
            resultMultiply = multiply_sertilp(matrix, vector=vector, align=align, slice_size=ss, threads_per_row=tpr, prefetch=prefetch, repeat=repeat)
            resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, output=output, formatName='sertilp', compensate=com, matrixName=matrixPath, sep=sep, eol=eol)
            if test: testResult(resultNumpy, resultMultiply[0], test, quite)
        if ert:
            if not quite: click.secho(getMessage('multiply_ertilp'), fg=colors['warning'])
            resultMultiply = multiply_ertilp(matrix, vector=vector, block_size=block, threads_per_row=tpr, prefetch=prefetch, repeat=repeat)
            resumeResult(resultMuliply=resultMultiply, resultPrint=result, timePrint=time, avrTimePrint=avrtime, stdTimePrint=std, quite=quite, output=output, formatName='ertilp', compensate=com, matrixName=matrixPath, sep=sep, eol=eol)
            if test: testResult(resultNumpy, resultMultiply[0], test, quite)
             
def resumeResult(resultMuliply, resultPrint, timePrint, avrTimePrint, stdTimePrint, quite, output, formatName, compensate, matrixName, sep, eol):
    times = resultMuliply[1]
    if compensate:
        times = times[compensate:]
    if resultPrint:
        click.echo(('' if quite else getMessage('result')) + stringVector(resultMuliply[0]))
    if timePrint:
        click.echo(('' if quite else getMessage('timeList')) + str(times))
    if avrTimePrint:
        click.echo(('' if quite else getMessage('avrTime')) + str(avr(times)))
    if stdTimePrint:
        click.echo(('' if quite else getMessage('stdTime')) + str(nstd(times)))
    if output:
        avrTime = str(avr(times))
        stdTime = str(nstd(times))
        data = [matrixName, formatName, avrTime, stdTime]
        data.extend(map(str, times))
        output.write(sep.join(data) + eol )
        
def testResult(model, check, confidenceFactor, quite):
    string = [('' if quite else getMessage('test')) ]
    vectorString = stringVector(
                        map(str, resultEquals(model, check, confidenceFactor)), \
                        valueFormat='%s', \
                        width=100, \
                        rowFormat='  {0:<7}{1:<}'
                    )
    if vectorString.lstrip(): 
        string.append(vectorString)
        click.echo('\n'.join(string))
def getMessage(idMessage):
    return {
        'error' : 'error',
        'title' : 'Process matrix: ',
        'pm' : 'Representation of data matrix: ',
        'conv' : u'Representations of the matrix after conversion to selected formats: ',
        'convEll' : 'Conversion to ELLPACK',
        'convSliced' : 'Conversion to SLICED ELLPACK',
        'convSertilp' : 'Conversion to SERTILP',
        'convErtilp' : 'Conversion to ERTILP',
        'multiply' : u'Multiply matrix %s by the vector %s',
        'multiplyCpu' : u'Multiplication with Numpy (only CPU)',
        'multiplyEll' : u'Multiplication with ELLPACK',
        'multiply_sertilp' : u'Multiplication with SERTILP',
        'multiplySliced' : u'Multiplication with SLICED',
        'multiply_ertilp' : u'Multiplication with ERTILP',
        'multiply_csr' : 'Multiplication with CSR',
        'result' : u'Result: ',
        'timeList' : u'List of times multiplication [ms]: ',
        'avrTime' : u'Average time [ms]: ',
        'test': 'Errors (position, different, relative error): ',
        'info_rows': 'Rows: ',
        'info_cols': 'Cols: ',
        'info_nnz': 'NNZ: ',
        'info_sparse': 'Sparsing: ',
        'info_title': 'Info about matrix: ',
        'vec': 'Representation of data vector: ',
        'paramInfo': 'Parameters for multiplication: ',
        'open_failed': "File %s open failed.",
        'bad_length': 'Does not exist a vector with length equal to number of columns of matrix: %s.',
        'stdTime': 'Standard deviation: '
    }.get(idMessage, 'error')
          
if __name__ == '__main__':
    cli()
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 19:03:24 2014

@author: Sławomir Figiel
"""

import click
import scipy.io
from matrixFormat import convertToELL, convertToErtilp, convertToSertilpELL, convertToSlicedELL
from filesUtilites import sortPaths, pathReduction

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

@click.option('-ell', '--ellpack', 'ell', is_flag=True, help='Use Ellpack format')
@click.option('-sle', '--sliced', 'sle', is_flag=True, help='Use Sliced Ellpack format')
@click.option('-see', '--sertilp', 'see', is_flag=True, help='Use Sertilp Ellpack format')
@click.option('-ert', '--ertilp', 'ert', is_flag=True, help='Use Ertilp format')
@click.argument('matrix-paths', nargs=-1, required=True, type=click.Path(exists=True))
def cli(block, ss, tpr, align, prefetch, ell, sle, see, ert, matrix_paths):
    paths = map(str, matrix_paths)
    matrices = sortPaths(pathReduction(paths), '.mtx')['.mtx']
    for matrixPath in matrices:
        click.secho(getMessage('conv'), fg=colors['success'])
        try:
            matrix = scipy.io.mmread(matrixPath)
        except:
            click.secho(getMessage('open_failed') % matrixPath, fg=colors['danger'])
            continue
        if ell:
            click.secho(getMessage('convEll'), fg=colors['warning'])
            printFormat(convertToELL(matrix, array=False))
        if sle:
            click.secho(getMessage('convSliced'), fg=colors['warning'])
            printFormat(convertToSlicedELL(matrix, array=False, watkiNaWiersz=tpr, sliceSize=ss, align=align))
        if see:
            click.secho(getMessage('convSertilp'), fg=colors['warning'])
            printFormat(convertToSertilpELL(matrix, array=False, watkiNaWiersz=tpr, sliceSize=ss, align=align, prefetch=prefetch))
        if ert:
            click.secho(getMessage('convErtilp'), fg=colors['warning'])
            printFormat(convertToErtilp(matrix, threadPerRow=tpr, prefetch=prefetch, array=False))

def printFormat(convertedMatrix):
    formatFirstRow = '{0:<7}{1:>18}{2:>18}'
    formatSecondRow = '{0:<7}{1:>12}'
    formatTable = '{0:<50}{1:>}'
    rows = zip(convertedMatrix[0], convertedMatrix[1])
    first = [click.style(formatFirstRow.format('Idx', 'Value', 'Column'), fg=colors['info']), ]
    for idx, val in enumerate(rows):
        if val[0]:
            first.append(formatFirstRow.format(idx, val[0], val[1]))
    second = [click.style(formatSecondRow.format('       Row', '      Row length'), fg=colors['info'])]     
    for idx, val in enumerate(convertedMatrix[2]):
        if val:
            second.append(formatSecondRow.format(idx, val))
        
    if len(convertedMatrix) > 3:
            second.append('')
            second.append(click.style(formatSecondRow.format('Slice', 'Slice size'), fg=colors['info']))
            for idx, val in enumerate(convertedMatrix[3]):
                second.append(formatSecondRow.format(idx, val))
                
    lenFirst = len(first)
    lenSecond = len(second)
    if lenFirst > lenSecond:
        second += [''] * (lenFirst - lenSecond)
    else:
        first += [''] * (lenSecond - lenFirst)
    
    table = zip(first, second)
    for left, right in table:
        click.echo(formatTable.format(left, right))
    
        
        
        
def getMessage(idMessage):
    return {'conv' : u'Representations of the matrix after conversion to selected formats: ',
            'convEll' : 'Conversion to ELLPACK',
            'convSliced' : 'Conversion to SLICED ELLPACK',
            'convSertilp' : 'Conversion to SERTILP',
            'convErtilp' : 'Conversion to ERTILP',
            'open_failed': "File %s open failed."
    }.get(idMessage, 'error')
        
if __name__ == '__main__':
    cli()
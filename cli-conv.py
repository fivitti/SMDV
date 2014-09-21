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
@click.option('-q', '--quite', is_flag=True, help='Without messages, only effects functions.')

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
def cli(block, ss, tpr, align, prefetch, ell, sle, see, ert, quite, matrix_paths):
    paths = map(str, matrix_paths)
    matrices = sortPaths(pathReduction(paths), '.mtx')['.mtx']
    lang = 'en'
    for matrixPath in matrices:
        if not quite: click.secho(getMessage('conv', lang), fg=colors['success'])
        try:
            matrix = scipy.io.mmread(matrixPath)
        except:
            click.secho(getMessage('open_failed') % matrixPath, fg=colors['danger'])
            continue
        stringListInList = lambda listInList: '\n'.join(map(str, listInList))
        if ell:
            if not quite: click.secho(getMessage('convEll', lang), fg=colors['warning'])
            click.echo(stringListInList(convertToELL(matrix, array=False)))
        if sle:
            if not quite: click.secho(getMessage('convSliced', lang), fg=colors['warning'])
            click.echo(stringListInList(convertToSlicedELL(matrix, array=False, watkiNaWiersz=tpr, sliceSize=ss, align=align)))
        if see:
            if not quite: click.secho(getMessage('convSertilp', lang), fg=colors['warning'])
            click.echo(stringListInList(convertToSertilpELL(matrix, array=False, watkiNaWiersz=tpr, sliceSize=ss, align=align, prefetch=prefetch)))
        if ert:
            if not quite: click.secho(getMessage('convErtilp', lang), fg=colors['warning'])
            click.echo(stringListInList(convertToErtilp(matrix, threadPerRow=tpr, prefetch=prefetch, array=False)))

def getMessage(idMessage, lang='en'):
    return {'conv' : u'Representations of the matrix after conversion to selected formats: ',
            'convEll' : 'Conversion to ELLPACK',
            'convSliced' : 'Conversion to SLICED ELLPACK',
            'convSertilp' : 'Conversion to SERTILP',
            'convErtilp' : 'Conversion to ERTILP',
            'open_failed': "File %s open failed."
    }.get(idMessage, 'error')
        
if __name__ == '__main__':
    cli()
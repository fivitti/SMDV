# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 15:01:37 2014

@author: Sławomir Figiel
"""
import click
from numpy import load
import scipy.io
from matrixUtilites import twoColumnString, getInfoMatrix, getInfoVector, stringVector
from filesUtilites import sortPaths, pathReduction

colors = {
        'success' : 'green',
        'info' : 'cyan',
        'warning' : 'yellow',
        'danger' : 'red'
    }
    
@click.command()
@click.option('-i/-no-i', '--info/--no-info', 'info_print', default=True, help='Display parameters. Default: True.')
@click.option('-r', '--representation', is_flag=True, help='Displays a representation of the data in the matrix.')
@click.option('-s/-d', '--sparse/--dense', default=True, help='Choose a display style matrix. Default sprase from Scipy. Dense from Numpy.todense()')
@click.option('-sep', '--separator', 'sep',  default='; ', help='Separator for data in report. Default: "; "')
#@click.option('-eol', '--end-of-line', 'eol_ch', default='lin', type=click.Choice(['win', 'lin']), help=r'End of line for data in report. Default Linux style: "\n".')
@click.option('-o', '--output', type=click.File(mode='a', lazy=True), help='File to save raport. Format CSV. If exist append new data. Added to the file it info if it is created.')
@click.argument('paths', type=click.Path(exists=True), nargs=-1)
def cli(info_print, representation, sparse, sep, output, paths):
    eol = '\n'
    paths = map(str, paths)
    headersMatrix = ['matrix', 'rows', 'cols', 'nnz', 'sparsing [%]']
    headersVector = ['vector', 'length', 'nnz', 'sparsing [%]']
    dictFile = sortPaths(pathReduction(paths), '.mtx', '.npy')
    matrices = dictFile['.mtx']
    vectors = dictFile['.npy']
    if output and matrices:
        output.write(sep.join(headersMatrix))
        output.write(eol)
    for matrixPath in matrices:
        click.secho(getMessage('info_title') + matrixPath, fg=colors['success'])   
        try:
            matrix = scipy.io.mmread(matrixPath)
        except:
            click.secho(getMessage('open_failed') % matrixPath, fg=colors['danger'])
            continue
        info = map(str, getInfoMatrix(matrix))
        if info_print:
            click.secho(getMessage('info_title_matrix'), fg=colors['info'])
            click.echo(twoColumnString(headersMatrix[1:], info))
        if output:
            output.write(matrixPath + sep + sep.join(info))
            output.write(eol)
        if representation:
            click.secho(getMessage('mat'), fg=colors['info'])
            if sparse:
                click.echo(str(matrix))
            else:
                click.echo(str(matrix.todense()))
            
    if output and vectors:
        output.write(sep.join(headersVector) + eol)
    for vectorPath in vectors:
        click.secho(getMessage('info_title') + vectorPath, fg=colors['success']) 
        try:
            vector = load(vectorPath)
        except:
            click.secho(getMessage('open_failed') % vectorPath, fg=colors['danger'])
            continue
        info = map(str, getInfoVector(vector))
        if info_print:
            click.secho(getMessage('info_title_vector'), fg=colors['info'])
            click.echo(twoColumnString(headersVector[1:], info))
        if output:
            output.write(vectorPath + sep + sep.join(info) + eol) 
        if representation:
            click.secho(getMessage('vec'), fg=colors['info'])
            click.echo(stringVector(vector))


def getMessage(idMessage):
    return {'info_rows': 'Rows: ',
            'info_cols': 'Cols: ',
            'info_nnz': 'NNZ: ',
            'info_length': 'Length: ',
            'info_sparse': 'Sparsing: ',
            'info_title': 'Process: ',
            'info_title_matrix': 'Info about matrix: ',
            'info_title_vector': 'Info about vector: ',
            'vec': 'Representation of data vector: ',
            'mat': 'Representation of data matrix: ',
            'open_failed': "File %s open failed."
    }.get(idMessage, 'error')

if __name__ == '__main__':
    cli()
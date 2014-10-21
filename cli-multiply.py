# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 13:47:34 2014

@author: Slawomir Figiel
"""

import click
import scipy.io
from os.path import isfile
from numpy import average as avr, std as nstd, load
from matrixmultiplication import multiply_cpu, multiply_ellpack, \
                                 multiply_sliced, multiply_sertilp, \
                                 multiply_ertilp, multiply_csr
from matrixutilites import string_vector, result_equals, dict_vector_paths
from filesutilites import path_reduction, sort_paths

COLORS = {'success' : 'green',
          'info' : 'cyan',
          'danger' : 'red',
          'warning' : 'yellow'
         }

@click.command()
@click.option('-b', '--block', default=128,
              help='Block size for CUDA. Default: 128')
@click.option('-ss', '--slice-size', 'ss', default=64,
              help='Slice size for ...Ellpack. Default: 64')
@click.option('-tpr', '--thread-per-row', 'tpr', default=2,
              help='Thread per row. Default: 2')
@click.option('-a', '--align', default=32,
              help='Align const for Ellpack. Default: 32')
@click.option('-p', '--prefetch', default=2,
              help='PreFetch for SlicedEllpack. Default: 2')
@click.option('-r', '--repeat', default=1,
              help='Count of repetitions calculations. Deafult: 1')

@click.option('-csr', '--csr', 'csr', is_flag=True, help='Use CSR format')
@click.option('-ell', '--ellpack', 'ell', is_flag=True,
              help='Use Ellpack format')
@click.option('-sle', '--sliced', 'sle', is_flag=True,
              help='Use Sliced Ellpack format')
@click.option('-see', '--sertilp', 'see', is_flag=True,
              help='Use Sertilp Ellpack format')
@click.option('-ert', '--ertilp', 'ert', is_flag=True,
              help='Use Ertilp format')
@click.option('-cpu', '--cpu', 'cpu', is_flag=True,
              help='Use CPU method multiplication (without GPU) on Numpy')

@click.option('-rst', '--result', is_flag=True,
              help='Print result multiplication.')
@click.option('-t', '--time', is_flag=True,
              help='Print list of time multiplication')
@click.option('-avr', '--avrtime', is_flag=True,
              help='Print average time multiplication')
@click.option('-std', '--standard-deviation', 'std', is_flag=True,
              help='Print standard deviation of time multiplication')
@click.option('--test', type=click.FLOAT,
              help='Testing result multiplication. Print bad row. Value is '
                   'confidence factor.')
@click.option('-com', '--compensate', 'com', type=click.INT,
              help='N first time are remove (returned times decremented '
                   'by n). Recommended in testing the speed, because the n '
                   'first times (e. g. one) are a long delay.')
@click.option('-o', '--output', type=click.File(mode='a', lazy=True),
              help='File to save raport. Format CSV. If exist append new '
                   'data. Added to the file it info if it is created.')
@click.option('-param', '--parameters', is_flag=True,
              help='Print value of parameters.')
@click.option('-sep', '--separator', 'sep', default='; ',
              help='Separator for data in report. Default: "; "')

@click.argument('vector-path', nargs=1, required=True,
                type=click.Path(exists=True))
@click.argument('matrix-path', nargs=1, required=True,
                type=click.Path(exists=True))
def cli(block, ss, tpr, align, prefetch, csr, ell, sle, see, ert, cpu,
        repeat, result, time, avrtime, std, test, com, output, parameters,
        vector_path, sep, matrix_path):
    '''
    Command line interface for test execute matrix multiplication.
    '''
    eol = '\n'
    param = {'Block' : str(block),
             'Slice size' : str(ss),
             'Threads per row' : str(tpr),
             'Align' : str(align),
             'Prefetch' : str(prefetch),
             'Repeat' : str(repeat),
             'Compensate' : str(com) if com else '0'
            }
    vectors_dict = dict_vector_paths(
                    sort_paths(path_reduction(
                        [str(vector_path),]), '.npy')
                    ['.npy'])
    matrices_paths = sort_paths(
                        path_reduction(
                            [str(matrix_path),]),
                            '.mtx')['.mtx']

    if com:
        repeat += com
    #Print parameters
    if parameters:
        click.secho(_get_msg('paramInfo'), fg=COLORS['info'])
        param_rows = []
        for key, value in param.items():
            param_rows.append('  {0:<20}{1}'.format(key, value))
        click.echo('\n'.join(param_rows))
    #Create file and add headers
    if output and not isfile(output.name):
        output.write(sep.join(param.keys()) + eol)
        output.write(sep.join(param.values()) + eol)
        output.write(eol)
        headers = ['matrix', 'format', 'average time',
                   'standard deviation time', 'times']
        output.write(sep.join(headers) + eol)

    for matrix_path in matrices_paths:
        try:
            matrix = scipy.io.mmread(matrix_path)
        except:
            click.secho(_get_msg('open_failed') % matrix_path,
                        fg=COLORS['danger'])
            continue
        #Find vector to matrix
        cols = matrix.shape[1]
        if not cols in vectors_dict.keys():
            click.secho(_get_msg('bad_length') % matrix_path,
                        fg=COLORS['danger'])
            continue
        vector_path = vectors_dict[cols]
        click.secho(_get_msg('multiply') % (matrix_path, vector_path),
                    fg=COLORS['success'])
        vector = load(vector_path)
        result_numpy = ''
        #Multiplication
        if cpu:
            click.secho(_get_msg('multiplyCpu'), fg=COLORS['warning'])
            result_multiply = multiply_cpu(matrix, repeat=repeat,
                                           vector=vector)
            if test:
                result_numpy = result_multiply[0]
            _resume_result(result_multiply=result_multiply,
                           result_print=result, time_print=time,
                           avr_time_print=avrtime, std_time_print=std,
                           output=output, format_name='cpu', compensate=com,
                           matrix_name=matrix_path, sep=sep, eol=eol)
        elif test:
            result_numpy = multiply_cpu(matrix, vector=vector,
                                        repeat=repeat)[0]
        if csr:
            click.secho(_get_msg('multiply_csr'), fg=COLORS['warning'])
            result_multiply = multiply_csr(matrix, vector=vector,
                                           repeat=repeat, block_size=block)
            _resume_result(result_multiply=result_multiply,
                           result_print=result, time_print=time,
                           avr_time_print=avrtime, std_time_print=std,
                           output=output, format_name='csr', compensate=com,
                           matrix_name=matrix_path, sep=sep, eol=eol)
            if test:
                _test_result(result_numpy, result_multiply[0], test)
        if ell:
            click.secho(_get_msg('multiplyEll'), fg=COLORS['warning'])
            result_multiply = multiply_ellpack(matrix, vector=vector,
                                               repeat=repeat, block_size=block)
            _resume_result(result_multiply=result_multiply,
                           result_print=result, time_print=time,
                           avr_time_print=avrtime, std_time_print=std,
                           output=output, format_name='ellpack',
                           compensate=com, matrix_name=matrix_path,
                           sep=sep, eol=eol)
            if test:
                _test_result(result_numpy, result_multiply[0], test)
        if sle:
            click.secho(_get_msg('multiplySliced'), fg=COLORS['warning'])
            result_multiply = multiply_sliced(matrix, vector=vector,
                                              align=align, slice_size=ss,
                                              threads_per_row=tpr,
                                              repeat=repeat)
            _resume_result(result_multiply=result_multiply,
                           result_print=result, time_print=time,
                           avr_time_print=avrtime, std_time_print=std,
                           output=output, format_name='sliced',
                           compensate=com, matrix_name=matrix_path, sep=sep,
                           eol=eol)
            if test:
                _test_result(result_numpy, result_multiply[0], test)
        if see:
            click.secho(_get_msg('multiply_sertilp'), fg=COLORS['warning'])
            result_multiply = multiply_sertilp(matrix, vector=vector,
                                               align=align, slice_size=ss,
                                               threads_per_row=tpr,
                                               prefetch=prefetch,
                                               repeat=repeat)
            _resume_result(result_multiply=result_multiply,
                           result_print=result, time_print=time,
                           avr_time_print=avrtime, std_time_print=std,
                           output=output, format_name='sertilp',
                           compensate=com, matrix_name=matrix_path,
                           sep=sep, eol=eol)
            if test:
                _test_result(result_numpy, result_multiply[0], test)
        if ert:
            click.secho(_get_msg('multiply_ertilp'), fg=COLORS['warning'])
            result_multiply = multiply_ertilp(matrix, vector=vector,
                                              block_size=block,
                                              threads_per_row=tpr,
                                              prefetch=prefetch, repeat=repeat)
            _resume_result(result_multiply=result_multiply,
                           result_print=result, time_print=time,
                           avr_time_print=avrtime, std_time_print=std,
                           output=output, format_name='ertilp',
                           compensate=com, matrix_name=matrix_path,
                           sep=sep, eol=eol)
            if test:
                _test_result(result_numpy, result_multiply[0], test)

def _resume_result(result_multiply, result_print, time_print, avr_time_print,
                   std_time_print, output, format_name, compensate,
                   matrix_name, sep, eol):
    '''
    Method generalized processing resume result multiplication.
    '''
    times = result_multiply[1]
    if compensate:
        times = times[compensate:]
    if result_print:
        click.echo(_get_msg('result') + string_vector(result_multiply[0]))
    if time_print:
        click.echo(_get_msg('timeList') + str(times))
    if avr_time_print:
        click.echo(_get_msg('avr_time') + str(avr(times)))
    if std_time_print:
        click.echo(_get_msg('std_time') + str(nstd(times)))
    if output:
        avr_time = str(avr(times))
        std_time = str(nstd(times))
        data = [matrix_name, format_name, avr_time, std_time]
        data.extend(map(str, times))
        output.write(sep.join(data) + eol)

def _test_result(model, check, confidence_factor):
    '''Method equal vectors and print result if errors.'''
    string = [_get_msg('test'), ]
    vector_string = string_vector(
                        map(str, result_equals(model, check,
                                               confidence_factor)),
                        value_format='%s',
                        width=100,
                        row_format='  {0:<7}{1:<}'
                    )
    if vector_string.lstrip():
        string.append(vector_string)
        click.echo('\n'.join(string))
def _get_msg(id_message):
    ''' Method return message to be printed on console. '''
    return {'error' : 'error',
            'title' : 'Process matrix: ',
            'multiply' : u'Multiply matrix %s by the vector %s',
            'multiplyCpu' : u'Multiplication with Numpy (only CPU)',
            'multiplyEll' : u'Multiplication with ELLPACK',
            'multiply_sertilp' : u'Multiplication with SERTILP',
            'multiplySliced' : u'Multiplication with SLICED',
            'multiply_ertilp' : u'Multiplication with ERTILP',
            'multiply_csr' : 'Multiplication with CSR',
            'result' : u'Result: ',
            'timeList' : u'List of times multiplication [ms]: ',
            'avr_time' : u'Average time [ms]: ',
            'test': 'Errors (position, different, relative error): ',
            'vec': 'Representation of data vector: ',
            'paramInfo': 'Parameters for multiplication: ',
            'open_failed': "File %s open failed.",
            'bad_length': 'Does not exist a vector with length equal to '
                          'number of columns of matrix: %s.',
            'std_time': 'Standard deviation: '
           }.get(id_message, 'error')

if __name__ == '__main__':
    cli()
    
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 19:03:24 2014

@author: Slawomir Figiel
"""

import click
import scipy.io
from matrixformat import convert_to_ellpack, convert_to_ertilp, \
                         convert_to_sertilp, convert_to_sliced
from filesutilites import sort_paths, path_reduction

COLORS = {'success' : 'green',
          'info' : 'cyan',
          'danger' : 'red',
          'warning' : 'yellow'
         }

@click.command()

@click.option('-slice_size', '--slice-size', 'slice_size', default=64,
              help='Slice size for ...Ellpack. Default: 64')
@click.option('-tpr', '--thread-per-row', 'tpr', default=2,
              help='Thread per row. Default: 2')
@click.option('-a', '--align', default=32,
              help='Align const for Ellpack. Default: 32')
@click.option('-p', '--prefetch', default=2,
              help='PreFetch for SlicedEllpack. Default: 2')

@click.option('-ell', '--ellpack', 'ell', is_flag=True,
              help='Use Ellpack format')
@click.option('-sle', '--sliced', 'sle', is_flag=True,
              help='Use Sliced Ellpack format')
@click.option('-see', '--sertilp', 'see', is_flag=True,
              help='Use Sertilp Ellpack format')
@click.option('-ert', '--ertilp', 'ert', is_flag=True,
              help='Use Ertilp format')
@click.argument('matrix-paths', nargs=-1, required=True,
                type=click.Path(exists=True))
def cli(slice_size, tpr, align, prefetch, ell, sle, see, ert,
        matrix_paths):
    '''
    Command line interface for present result convertion of matrices.

    Supported formats: Ellpack, Sliced, Sertilp, Ertilp.
    '''
    paths = [str(i) for i in matrix_paths]
    matrices = sort_paths(path_reduction(paths), '.mtx')['.mtx']
    for matrix_path in matrices:
        click.secho(_get_msg('conv'), fg=COLORS['success'])
        try:
            matrix = scipy.io.mmread(matrix_path)
        except:
            click.secho(_get_msg('open_failed') % matrix_path,
                        fg=COLORS['danger'])
            continue
        if ell:
            click.secho(_get_msg('convEll'), fg=COLORS['warning'])
            _print_format(convert_to_ellpack(matrix, array=False))
        if sle:
            click.secho(_get_msg('convSliced'), fg=COLORS['warning'])
            _print_format(convert_to_sliced(matrix, threads_per_row=tpr,
                                            slice_size=slice_size, align=align,
                                            array=False))
        if see:
            click.secho(_get_msg('convSertilp'), fg=COLORS['warning'])
            _print_format(convert_to_sertilp(matrix, array=False,
                                             threads_per_row=tpr,
                                             slice_size=slice_size,
                                             align=align,
                                             prefetch=prefetch))
        if ert:
            click.secho(_get_msg('convErtilp'), fg=COLORS['warning'])
            _print_format(convert_to_ertilp(matrix, prefetch=prefetch,
                                            threads_per_row=tpr, array=False))
def _print_format(converted_matrix):
    '''
    Method print converted matrix on console.
    '''
    #Define format columns
    format_first_col = '{0:<7}{1:>18}{2:>18}'
    format_second_col = '{0:<7}{1:>12}'
    #Define format table
    format_table = '{0:<50}{1:>}'
    #Join vector values and columns
    rows = zip(converted_matrix[0], converted_matrix[1])
    #Set headers first column
    first = [click.style(format_first_col.format('Idx', 'Value', 'Column'),
                         fg=COLORS['info']), ]
    #Process first column
    for idx, val in enumerate(rows):
        if val[0]:
            first.append(format_first_col.format(idx, val[0], val[1]))
    #Set headers second column
    second = [click.style(format_second_col.format('       Row',
                                                   '      Row length'),
                          fg=COLORS['info'])]
    #Process second column
    for idx, val in enumerate(converted_matrix[2]):
        if val:
            second.append(format_second_col.format(idx, val))

    #Add part with slice start vector
    if len(converted_matrix) > 3:
        second.append('')
        second.append(click.style(format_second_col.format('Slice',
                                                           'Slice size'),
                                  fg=COLORS['info']))
        for idx, val in enumerate(converted_matrix[3]):
            second.append(format_second_col.format(idx, val))

    #Normalize length first and second column
    len_first = len(first)
    len_second = len(second)
    if len_first > len_second:
        second += [''] * (len_first - len_second)
    else:
        first += [''] * (len_second - len_first)

    #Join columns and print on console
    table = zip(first, second)
    for left, right in table:
        click.echo_via_pager(format_table.format(left, right))

def _get_msg(id_message):
    ''' Method return message to be printed on console. '''
    return {'conv' : 'Representations of the matrix after conversion'
                     'to selected formats: ',
            'convEll' : 'Conversion to ELLPACK',
            'convSliced' : 'Conversion to SLICED ELLPACK',
            'convSertilp' : 'Conversion to SERTILP',
            'convErtilp' : 'Conversion to ERTILP',
            'open_failed': "File %s open failed."
           }.get(id_message, 'error')

if __name__ == '__main__':
    cli()

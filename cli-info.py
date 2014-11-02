# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 15:01:37 2014
@author: Slawomir Figiel
"""
import click
import scipy.io
from numpy import load
from matrixutilites import two_column_string, get_info_matrix, \
                           get_info_vector, \
                           string_vector
from filesutilites import sort_paths, path_reduction

COLORS = {'success' : 'green',
          'info' : 'cyan',
          'warning' : 'yellow',
          'danger' : 'red'
         }

@click.command()
@click.option('-i/-no-i', '--info/--no-info', 'info_print',
              default=True, help='Display parameters. Default: True.')
@click.option('-r', '--representation', is_flag=True,
              help='Displays a representation of the data in the matrix.')
@click.option('-sep', '--separator', 'sep', default='; ',
              help='Separator for data in report. Default: "; "')
@click.option('-o', '--output', type=click.File(mode='a', lazy=True),
              help='File to save raport. Format CSV. If exist append new '
                   'data. Added to the file it info if it is created.')
@click.argument('paths', type=click.Path(exists=True), nargs=-1)
def cli(info_print, representation, sep, output, paths):
    '''
    Commnad Line Interface for getting information about matrices and vectors.

    \b
    Supported matrix format: MatrixMarket Format [.mtx].
    Supported vector formats: Numpy Binary File [.npy]
    '''
    eol = '\n'
    paths = [str(i) for i in paths]
    dict_file = sort_paths(path_reduction(paths), '.mtx', '.npy')
    matrix_dict = {'headers' : ['matrix', 'rows', 'cols',
                                'nnz', 'sparsing [%]'],
                   'load_method' : scipy.io.mmread,
                   'type'  : 'matrix',
                   'files' : dict_file['.mtx'],
                   'info_method' : get_info_matrix,
                   'format_method' : str
                  }
    vector_dict = {'headers' : ['vector', 'length', 'nnz', 'sparsing [%]'],
                   'load_method' : load,
                   'type' : 'vector',
                   'files' : dict_file['.npy'],
                   'info_method' : get_info_vector,
                   'format_method' : string_vector
                  }
    _process_files(output, info_print, representation, sep, eol,
                   param_dict=matrix_dict)
    _process_files(output, info_print, representation, sep, eol,
                   param_dict=vector_dict)


def _process_files(output, info_print, representation, sep, eol, param_dict):
    '''
    Method generalized processing matrices and vectors

    Parameters
    ==========
    output : bool
        if true save information to file
    info_print : bool
        if true print information in console
    sep : string
        separator in file
    eol : string
        end of line char in file
    param_dict : dictionary
        Dictionary with parameters. Keys:
        headers, load_method, type, info_method, format_method

    Return
    ======
    Nothing. Method print on console and write to file.
    '''
    if output and param_dict['files']:
        output.write(sep.join(param_dict['headers']) + eol)
    for path in param_dict['files']:
        click.secho(_get_msg('info_title') + path, fg=COLORS['success'])
        try:
            object_ = param_dict['load_method'](path)
        except:
            click.secho(_get_msg('open_failed') % path,
                        fg=COLORS['danger'])
            continue
        info = [str(i) for i in param_dict['info_method'](object_)]
        if info_print:
            click.secho(_get_msg('info_title_type') % param_dict['type'],
                        fg=COLORS['info'])
            click.echo(two_column_string(param_dict['headers'][1:], info))
        if output:
            output.write(path + sep + sep.join(info) + eol)
        if representation:
            click.secho(_get_msg('repr') % param_dict['type'],
                        fg=COLORS['info'])
            click.echo(param_dict['format_method'](object_))

def _get_msg(id_message):
    '''
    Method return message to be printed on console.
    '''
    return {'info_title': 'Process:',
            'info_title_type': 'Info about %s:',
            'repr': 'Representation of data %s:',
            'open_failed': "File %s open failed."
           }.get(id_message, 'error')

if __name__ == '__main__':
    cli()
    
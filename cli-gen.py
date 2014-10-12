# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 21:01:14 2014

@author: Sławomir Figiel
"""
import click
from matrixutilites import generate_vector, generate_sparse_matrix, string_vector, save_matrix_to_file, save_vector_to_numpy_file
from os.path import join, isdir

colors = {
        'success' : 'green',
        'info' : 'cyan',
        'warning' : 'yellow',
        'danger' : 'red'
    }
    
@click.group(chain=True)
def cli():
    '''
    '''
    pass

@cli.command()
@click.option('-l', '--length', type=click.INT, multiple=True, help='The length of the vector. Each call will generate a vector.')
@click.option('-min', '--minimum', default=-10, help='Minimal value. Default: -10.')
@click.option('-max', '--maximum', default=10, help='Maximum value. Default: 10.' )
@click.option('-i/-f', '--integer/--float', default=False, help='The draw integers or floating point. Default: floating.')
@click.option('-pc', '--percent', default=40.0, help='Percentage of zeroes in the vector. Default: 40%')
@click.option('-prec', '--precision', default=6, help='Precision float number. Default: 6')
@click.pass_context
def vector(ctx, length, minimum, maximum, integer, percent, precision):
    '''
    The command to generate vectors.
    '''
    vectors = []
    for l in length:   
        vec = generate_vector(length=l, minimum=minimum, maximum=maximum, integers=integer, percentage_zeros=percent, precision=precision, array=True)
        vectors.append(vec)
    ctx.obj['vectors'] = vectors
            
@cli.command()
@click.option('-s', '--shape', type=click.INT, nargs=2, multiple=True, help='The shape of the vector (rows cols). Each call will generate a matrix.')
@click.option('-min', '--minimum', default=-10, help='Minimal value. Default: -10.')
@click.option('-max', '--maximum', default=10, help='Maximum value. Default: 10.' )
@click.option('-i/-f', '--integer/--float', default=True, help='The draw integers or floating point. Default: floating.')
@click.option('-pc', '--percent', default=40.0, help='Percentage of zeroes in the vector.')
@click.option('-prec', '--precision', default=6, help='Precision float number.')
@click.pass_context
def matrix(ctx, shape, minimum, maximum, integer, percent, precision):
    '''
    The command to generate a matrix.
    '''
    matrices = []
    for s in shape:
        mat = generate_sparse_matrix(rows=s[0], cols=s[1], minimum=minimum, maximum=maximum, integers=integer, percentage_zeros=percent, precision=precision)
        matrices.append(mat)
    ctx.obj['matrices'] = matrices

@cli.command()
@click.option('-q', '--quite', is_flag=True, help='Without messages, only effects function')
@click.option('-nz', '--without-zeros', is_flag=True, help='Not write zeros in vectors')
@click.option('--dense/--no-dense', default=False, help='Display matrix as dense. Caution!')
@click.pass_context
def echo(ctx, quite, dense, without_zeros):
    '''
    Command to display the loaded and generated vectors and matrices. 
    Displays all objects stored in memory.
    '''
    if 'vectors' in ctx.obj:
        for vec in ctx.obj['vectors']:
            click.echo(click.style(('' if quite else 'Vector (len: %s): \n' % len(vec)), fg=colors['warning']) + string_vector(vec, without_zeros=without_zeros))
    if 'matrices' in ctx.obj:
        for mat in ctx.obj['matrices']:
            click.echo(click.style(('' if quite else 'Matrix (rows: %s, cols: %s): \n' % mat.shape), fg=colors['info']) + (str(mat) if not dense else str(mat.todense())))

@cli.command()
@click.option('-fm', '--subfolder-matrices', nargs=1, default='', type=click.STRING, help='Subfolder for saving matrices. (In FOLDER)')
@click.option('-fv', '--subfolder-vectors', nargs=1, default='', type=click.STRING, help='Subfolder for saving vectors. (In FOLDER)')
@click.option('-pm', '--prefix-matrices', default='Matrix', help='Prefix name for saving matrices. Default: "Matrix"')
@click.option('-pv', '--prefix-vectors', default='Vector', help='Prefix name for saving vectors. Default: "Vector"')
@click.option('-em', '--extension-matrices', default='mtx', help='Extension for saving matrices. Recomended: mtx')
@click.option('-ev', '--extension-vectors', default='npy', help='Extension for saving matrices. Recomended: npy')
@click.option('-sm, --suffix-matrices', 'suffix_matrices', default='', type=click.STRING, help='Suffix name for saving matrices. Will be added before the extension.')
@click.option('-sv', '--suffix-vectors', 'suffix_vectors', default='', type=click.STRING, help='Suffix name for saving vectors. Will be added before the extension.')
@click.option('-a', '--addition', 'addition', type=click.Choice(['dim', 'date', 'without' ]), default='dim', help='Choose a addition to the name: dimensions, date, without addition. Default: dimensions.')
@click.argument('folder', nargs=1, required=True, type=click.Path(exists=True))
@click.pass_context
def save(ctx, \
         subfolder_matrices, subfolder_vectors, \
         prefix_vectors, prefix_matrices, \
         extension_vectors, extension_matrices, \
         addition, \
         suffix_matrices, suffix_vectors, \
         folder):
    '''
    Command to save the loaded and generated vectors and matrices. 
    Saved all objects will be stored in memory.
    
    The argument FOLDER specifies folder where to save.
    
    Matrices will be saved to format MatrixMarketFile (.mtx).
    Vectors will be saved to format NumpyBinaryFile (.npy).
    '''
    if 'matrices' in ctx.obj:
        path = join(folder, subfolder_matrices, '')
        if not isdir(path):
            click.secho("Subfolder matrices don't exist. Save in FOLDER.", fg=colors['danger'])
            path = join(folder, '')
        for mat in ctx.obj['matrices']:
            save_matrix_to_file(matrix=mat, \
                             folder=path, \
                             prefix=prefix_matrices, \
                             extension=extension_matrices, \
                             date=(True if addition == 'date' else False), \
                             dimensions=(True if addition == 'dim' else False), \
                             suffix=suffix_matrices)
    if 'vectors' in ctx.obj:
        path = join(folder, subfolder_vectors, '')
        if not isdir(path):
            click.secho("Subfolder vectors don't exist. Save in FOLDER.", fg=colors['danger'])
            path = join(folder, '')
        for vec in ctx.obj['vectors']:
            save_vector_to_numpy_file(vector=vec, \
                                  folder=path, \
                                  prefix=prefix_vectors, \
                                  extension=extension_vectors, \
                                  date=(True if addition == 'date' else False), \
                                  length=(True if addition == 'dim' else False), \
                                  suffix=suffix_vectors)        

if __name__ == '__main__':
    cli(obj={})
    
    
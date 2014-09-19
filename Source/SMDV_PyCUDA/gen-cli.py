# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 21:01:14 2014

@author: HP
"""
import click

colors = {
        'success' : 'green',
        'info' : 'cyan',
        'warning' : 'yellow',
        'danger' : 'red'
    }
    
@click.group(chain=True)
def cli():
    pass

@cli.command()
@click.option('-l', '--length', type=click.INT, multiple=True, help='The length of the vector. Each call will generate a vector.')
@click.option('-min', '--minimum', default=-10, help='Minimal value. Default: -10.')
@click.option('-max', '--maximum', default=10, help='Maximum value. Default: 10.' )
@click.option('-i/-f', '--integer/--float', default=True, help='The draw integers or floating point. Default: floating.')
@click.option('-pc', '--percent', default=40.0, help='Percentage of zeroes in the vector.')
@click.option('-prec', '--precision', default=6, help='Precision float number.')
@click.pass_context
def vector(ctx, length, minimum, maximum, integer, percent, precision):
    from matrixUtilites import generateVector
    vectors = []
    for l in length:   
        vec = generateVector(length=l, minimum=minimum, maximum=maximum, integers=integer, percentageOfZeros=percent, precision=precision, array=True)
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
    from matrixUtilites import generateMatrixCsr
    matrices = []
    for s in shape:
        mat = generateMatrixCsr(rows=s[0], cols=s[1], minimum=minimum, maximum=maximum, integers=integer, percentageOfZeros=percent, precision=precision, mode=0)
        matrices.append(mat)
    ctx.obj['matrices'] = matrices

@cli.command()
@click.option('-q', '--quite', is_flag=True, help='Without messages, only effects function')
@click.option('--dense/--no-dense', default=False, help='Display matrix as dense. Caution!')
@click.pass_context
def echo(ctx, quite, dense):
    if 'vectors' in ctx.obj:
        for vec in ctx.obj['vectors']:
            click.echo(click.style(('' if quite else 'Vector (len: %s): \n' % len(vec)), fg=colors['info']) + str(vec))
    if 'matrices' in ctx.obj:
        for mat in ctx.obj['matrices']:
            click.echo(click.style(('' if quite else 'Matrix (rows: %s, cols: %s): \n' % mat.shape), fg=colors['info']) + (str(mat) if not dense else str(mat.todense())))
@cli.command()
@click.option('-fm', '--subfolder-matrices', nargs=1, default='', type=click.STRING)
@click.option('-fv', '--subfolder-vectors', nargs=1, default='', type=click.STRING)
@click.option('-pv', '--prefix-vectors', default='Vector_')
@click.option('-pm', '--prefix-matrices', default='Matrix_')
@click.option('-ev', '--extension-vectors', default='.npy')
@click.option('-em', '--extension-matrices', default='.mtx')
@click.option('-s/-d', '--shape/--date', 'shape', default=True)
@click.option('-sm, --suffix-matrices', 'suffix_matrices', default='', type=click.STRING)
@click.option('-sv', '--suffix-vectors', 'suffix_vectors', default='', type=click.STRING)
@click.argument('folder', nargs=1, required=True, type=click.Path(exists=True))
@click.pass_context
def save(ctx, subfolder_matrices, subfolder_vectors, prefix_vectors, prefix_matrices, extension_vectors, \
    extension_matrices, shape, suffix_matrices, suffix_vectors, folder):
    from os.path import join, isdir
    if 'matrices' in ctx.obj:
        from matrixUtilites import saveMatrixToFile
        path = join(folder, subfolder_matrices, '')
        if not isdir(path):
            click.secho("Subfolder matrices don't exist. Save in FOLDER.", fg=colors['danger'])
            path = join(folder, '')
        for mat in ctx.obj['matrices']:
            saveMatrixToFile(matrix=mat, folder=path, prefix=prefix_matrices, extension=extension_matrices, date=not shape, dimensions=shape, suffix=suffix_matrices)
    if 'vectors' in ctx.obj:
        from matrixUtilites import saveVectorToNumpyFile
        path = join(folder, subfolder_vectors, '')
        if not isdir(path):
            click.secho("Subfolder vectors don't exist. Save in FOLDER.", fg=colors['danger'])
            path = join(folder, '')
        for vec in ctx.obj['vectors']:
            saveVectorToNumpyFile(vector=vec, folder=path, prefix=prefix_vectors, extension=extension_vectors, date=not shape, length=shape, suffix=suffix_vectors)        
        


if __name__ == '__main__':
    cli(obj={})
    
    
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 21:01:14 2014

@author: Sławomir Figiel
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
    '''
    The command to generate vectors.
    '''
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
    '''
    The command to generate a matrix.
    '''
    from matrixUtilites import generateMatrixCsr
    matrices = []
    for s in shape:
        mat = generateMatrixCsr(rows=s[0], cols=s[1], minimum=minimum, maximum=maximum, integers=integer, percentageOfZeros=percent, precision=precision, mode=0)
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
        from matrixUtilites import stringVector
        for vec in ctx.obj['vectors']:
            click.echo(click.style(('' if quite else 'Vector (len: %s): \n' % len(vec)), fg=colors['warning']) + stringVector(vec, withoutZeros=without_zeros))
    if 'matrices' in ctx.obj:
        for mat in ctx.obj['matrices']:
            click.echo(click.style(('' if quite else 'Matrix (rows: %s, cols: %s): \n' % mat.shape), fg=colors['info']) + (str(mat) if not dense else str(mat.todense())))

@cli.command()
@click.option('-fm', '--subfolder-matrices', nargs=1, default='', type=click.STRING, help='Subfolder for saving matrices. (In FOLDER)')
@click.option('-fv', '--subfolder-vectors', nargs=1, default='', type=click.STRING, help='Subfolder for saving vectors. (In FOLDER)')
@click.option('-pm', '--prefix-matrices', default='Matrix_', help='Prefix name for saving matrices. Default: "Matrix_"')
@click.option('-pv', '--prefix-vectors', default='Vector_', help='Prefix name for saving vectors. Default: "Vector_"')
@click.option('-em', '--extension-matrices', default='.mtx', help='Extension for saving matrices. Recomended: .mtx')
@click.option('-ev', '--extension-vectors', default='.npy', help='Extension for saving matrices. Recomended: .npy')
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
    from os.path import join, isdir
    if 'matrices' in ctx.obj:
        from matrixUtilites import saveMatrixToFile
        path = join(folder, subfolder_matrices, '')
        if not isdir(path):
            click.secho("Subfolder matrices don't exist. Save in FOLDER.", fg=colors['danger'])
            path = join(folder, '')
        for mat in ctx.obj['matrices']:
            saveMatrixToFile(matrix=mat, \
                             folder=path, \
                             prefix=prefix_matrices, \
                             extension=extension_matrices, \
                             date=(True if addition == 'date' else False), \
                             dimensions=(True if addition == 'dim' else False), \
                             suffix=suffix_matrices)
    if 'vectors' in ctx.obj:
        from matrixUtilites import saveVectorToNumpyFile
        path = join(folder, subfolder_vectors, '')
        if not isdir(path):
            click.secho("Subfolder vectors don't exist. Save in FOLDER.", fg=colors['danger'])
            path = join(folder, '')
        for vec in ctx.obj['vectors']:
            saveVectorToNumpyFile(vector=vec, \
                                  folder=path, \
                                  prefix=prefix_vectors, \
                                  extension=extension_vectors, \
                                  date=(True if addition == 'date' else False), \
                                  length=(True if addition == 'dim' else False), \
                                  suffix=suffix_vectors)        
        
@cli.command()
@click.option('-v', '--vector', type=click.Path(exists=True), multiple=True, help='Path to the vector. Path to the vector. Can be called repeatedly.')
@click.option('-m', '--matrix', type=click.Path(exists=True), multiple=True, help='Path to the matrix. Can be called repeatedly.')
@click.option('-all', '--all-in-folder', type=click.Path(exists=True), help='The path to the folder from which all matrices and vectors are loaded.')
@click.pass_context
def load(ctx, vector, matrix, all_in_folder):
    '''
    The command loads the matrices and vectors of files. 
    Matrix supported formats: MatrixMarketFile (.mtx) 
    Vectors supported formats: NumpyBinaryFile (.npy) and other Numpy (experimental).
    '''
    matrix = list(matrix)
    vector = list(vector)
    if all_in_folder:
        from os import listdir
        from os.path import isfile, join
        for f in listdir(all_in_folder):
            f = join(all_in_folder, f)
            if not isfile(f):
                continue
            if f.endswith(".mtx"):
                matrix.append(f)
            elif f.endswith('.npy'):
                vector.append(f)
            else:
                from numpy import load as npLoad
                try:
                    npLoad(str(f))
                    vector.append(f)
                except IOError:
                    matrix.append(f)
    if len(matrix) > 0:
        import scipy.io
        if not 'matrices' in ctx.obj:
            ctx.obj['matrices'] = [] 
    for mat in matrix:
        m = scipy.io.mmread(str(mat))
        ctx.obj['matrices'].append(m)
        
    if len(vector) > 0:
        from numpy import load as npLoad
        if not 'vectors' in ctx.obj:
            ctx.obj['vectors'] = [] 
    for vec in vector:
        v = npLoad(str(vec))
        ctx.obj['vectors'].append(v)

if __name__ == '__main__':
    cli(obj={})
    
    
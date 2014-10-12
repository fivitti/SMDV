# -*- coding: utf-8 -*-
"""
Created on Sun May 04 20:45:18 2014
@author: Sławomir Figiel

Module provides the methods helpers to work in matrices and vectors:
    * Methods generated data
    * Methods get info about data
    * Methods build formatted string with data
    * Method verify data
    * Method sort data
"""
import scipy.io
import scipy.sparse
import numpy
import time
import os
from random import randint, uniform

def generate_sparse_matrix(rows, cols, minimum=1, maximum=100,
                           integers=True, percentage_zeros=40, precision=6):
    '''
    Method generate sparse matrix with specified parameters in format CSR.

    Parameters
    ==========
        cols : int > 0
            number of columns,
        rows : int > 0
            number of rows,
        minimum : int <> 0
            minimum non-zero value,
        maximum : int <> 0
            maximum non-zero value
        integers : boolean
            if true drawn only integers, else float
        percentage_zeros : int in range 0-100
            percentage of zeros in matrix
        precision : int > 0
            precision of random numbers

    Returns
    =======
    random matrix : scipy.sparse.csr_matrix
        Matrix of random values ​​with a certain percentage of zeros.

    Notes
    =====
    This method is intended to generate a sparse matrix.
    It may be inefficient for large dense matrix.

    Examples
    ========
    >>> A = generate_sparse_matrix(rows=3, cols=3, minimum=1, maximum=9,
                                   integers=False, percentage_zeros=33.3,
                                   precision=2)
    >>> print A
        (0, 0)	8.34
        (0, 1)	7.65
        (0, 2)	7.81
        (1, 1)	5.93
        (2, 0)	1.78
        (2, 1)	7.5
    '''
    to_draw = int(round(cols*rows - (cols*rows*percentage_zeros)/100))
    row_list = []
    col_list = []
    data = []
    points = set([])
    rand_method = randint if integers else uniform
    while to_draw > 0:
        row = randint(0, rows-1)
        col = randint(0, cols-1)
        point = (row, col)
        if not point in points:
            points.add(point)
            to_draw -= 1
    for i in points:
        row_list.append(i[0])
        col_list.append(i[1])
        value = round(rand_method(minimum, maximum), precision)
        while value == 0:
            value = round(rand_method(minimum, maximum), precision)
        data.append(value)

    return scipy.sparse.csr_matrix((data, (row_list, col_list)),
                                   shape=(rows, cols),
                                   dtype=numpy.float32)

def generate_vector(length, minimum=-1, maximum=1, integers=True,
                    percentage_zeros=40, precision=6, array=True):
    '''
    Method generate dense vector with specified parameters.

    Parameters
    ==========
        length : int > 0
            length of vector,
        minimum : int <> 0
            minimum non-zero value,
        maximum : int <> 0
            maximum non-zero value
        integers : boolean
            if true drawn only integers, else float
        percentage_zeros : int in range 0-100
            percentage of zeros in vector
        precision : int > 0
            precision of random numbers
        array : boolean
            if true return vector as numpy.array else standard python list

    Returns
    =======
    random vector : list or numpy.array (dtype=float32)
        Vector of random values ​​with a certain percentage of zeros.

    Notes
    =====
    This method is intended to generate a dense vector.
    It may be inefficient for large sparse vector.

    Examples
    ========
    >>> generate_vector(length=9, minimum=1, maximum=9,
                        integers=False, percentage_zeros=33.3,
                        precision=2, array=False)
        [0, 7.59, 0, 2.52, 5.08, 8.64, 1.75, 0, 1.18]
    '''
    if percentage_zeros > 100:
        percentage_zeros = 100
    elif percentage_zeros < 0:
        percentage_zeros = 0
    result = []
    zeros = round((length * percentage_zeros * 1.0) / 100)
    rand_method = randint if integers else uniform
    for _ in range(length):
        drawn = round(rand_method(minimum, maximum), precision)
        while drawn == 0:
            drawn = round(rand_method(minimum, maximum), precision)
        result.append(drawn)
    while zeros > 0:
        place = randint(0, length-1)
        if not result[place] == 0:
            result[place] = 0
            zeros -= 1
    if array:
        type_data = numpy.float32
        return numpy.array(result, dtype=type_data)
    else:
        return result

def save_vector_to_numpy_file(vector, folder='vectors\\', prefix='Vector',
                              extension='npy', date=False, length=True,
                              suffix=''):
    '''
    Method save vector to Numpy binary file. Generated filename and
    prevents overwriting an existing file.

    Parameters
    ==========
    vector : list or numpy.array
        vector to save (or different structure of Numpy
    folder : string
        directory to save. Must exist.
    prefix : string
        prefix filename
    extension : string
        extension file. Recommended '.npy'.
    date : boolean
        if true name will be appended date in format
        YearMonthDayHourMinuteSeconds
    length : boolean
        if true name will be appended length of vector
    suffix : boolean
        suffix filename

    Returns
    =======
    Nothing. Vector will be saved to disk in Numpy binary file.
    His name is: prefix_length_date_suffix_addition.extension
    Addition is adding only if file with this name exist.
    '''
    file_path = []
    extension = '.' + extension
    if prefix:
        file_path.append(prefix)
    if length:
        file_path.append(str(len(vector)))
    if date:
        date_string = ''
        now = time.localtime()
        for i in range(6):
            if now[i] < 10:
                date_string += '0'
            date_string += str(now[i])
        file_path.append(date_string)
    if suffix:
        file_path.append(suffix)
    if os.path.exists(os.path.join(folder, '_'.join(file_path)+extension)):
        addition = 1
        while os.path.exists(os.path.join(folder,
                                          '_'.join(file_path + \
                                                   [str(addition), ]) + \
                                          extension)):
            addition += 1
        file_path.append(str(addition))
    file_path = os.path.join(folder, '_'.join(file_path) + extension)
    numpy.save(file_path, vector)

def save_matrix_to_file(matrix, folder='matrices', prefix='Matrix',
                        extension='mtx', date=False, dimensions=True,
                        suffix=''):
    '''
    Method save matrix to Matrix Market CSR file. Generated filename and
    prevents overwriting an existing file.

    Parameters
    ==========
    vector : list or numpy.array
        vector to save (or different structure of Numpy
    folder : string
        directory to save. Must exist.
    prefix : string
        prefix filename
    extension : string
        extension file. Recommended '.npy'.
    date : boolean
        if true name will be appended date in format
        YearMonthDayHourMinuteSeconds
    dimensions : boolean
        if true name will be appended dimensions of matrix
    suffix : boolean
        suffix filename

    Returns
    =======
    Nothing. Matrix will be saved to disk in Matrix Market CSR file.
    His name is: prefix_length_date_suffix_addition.extension
    Addition is adding only if file with this name exist.
    '''
    filepath = []
    extension = '.' + extension
    if prefix:
        filepath.append(prefix)
    if dimensions:
        dimensions_list = []
        for i in matrix.shape:
            dimensions_list.append(str(i))
        filepath.append('x'.join(dimensions_list))
    if date:
        date_string = ''
        now = time.localtime()
        for i in range(6):
            if now[i] < 10:
                date_string += '0'
            date_string += str(now[i])
        filepath.append(date_string)
    if suffix:
        filepath.append(suffix)
    if os.path.exists(os.path.join(folder, '_'.join(filepath) + extension)):
        addition = 1
        while os.path.exists(os.path.join(folder,
                                          '_'.join(filepath + \
                                                   [str(addition),]) + \
                                          extension)):
            addition += 1
        filepath.append(str(addition))
    filepath = os.path.join(folder, '_'.join(filepath) + extension)
    scipy.io.mmwrite(str(filepath), matrix)

def _format_item(left, right, width=40, row_format='  {0:<7}{1:>9}'):
    '''
    Function helper for method string_vector and two_column_string.
    '''
    from textwrap import fill
    wrapped = fill(right, width=width, subsequent_indent=' '*15)
    return row_format.format(left, wrapped)

def string_vector(vector, without_zeros=False, value_format="%.15g",
                  width=40, row_format='  {0:<7}{1:>9}'):
    '''
    Method builds a legibly formatted vector string. It consists of two
    columns. First is the index position in the vector, and second value.

    Parameters
    ==========
    vector : enumerate
        vector, 1D list, numpy array, and other...
    without_zeros : boolean
        if true zero values ​​are not displayed
    value_format : string
        as will be formatted column values. Possible values ​​of
        the formatting strings. For float '%.15g' displays them with complete
        precision
    width : int
        width of values column
    row_format : string
        as will be formatted simple row. Possible values of string.format

    Returns
    =======
    formatted string : string

    Examples
    ========
    >>> vector = [5, 0, 3, 2, 0, 0, 0, 9, 10]
    >>> string_vector(vector, without_zeros=True)
      (0)            5
      (2)            3
      (3)            2
      (7)            9
      (8)           10
    '''
    result = []
    for idx, val in enumerate(vector):
        if without_zeros and not val:
            continue
        result.append(_format_item('(%s)' % str(idx), value_format % val,
                                   width=width, row_format=row_format))
    return '\n'.join(result)

def two_column_string(left_list, right_list, right_width=80,
                      row_format='  {0:<5}{1:>10}'):
    '''
    Method builds a legibly formatted string of two strings list.
    It consists of two columns. First is the element on first list,
    and second corresponding element on second list.

    Parameters
    ==========
    left_list : list of strings
        list of string. They will be in the left column
    right_list : list of strings
        list of string. They will be in the left column
    right_width : int
        width of right column
    row_format : string
        as will be formatted simple row. Possible values of string.format

    Returns
    =======
    formatted string : string

    Examples
    ========
    >>> headers = ['first', 'second', 'third', 'fourth', 'fifth']
    >>> values = ['lorem', 'ipsum', 'dolor', 'sit', 'amet']
        print two_column_string(headers,
                                values,
                                row_format='   {0:<10}{1:>10}')
            first          lorem
            second         ipsum
            third          dolor
            fourth           sit
            fifth           amet
    '''
    result = []
    for left, right in zip(left_list, right_list):
        result.append(_format_item('%s' % left,
                                   '%s' % right,
                                   width=right_width,
                                   row_format=row_format))
    return '\n'.join(result)

def get_info_matrix(matrix):
    '''
    Return basic information about matrix.

    Parameters
    ==========
    matrix : Scipy sparse matrix
        matrix

    Returns
    =======
    information about matrix : tuple
        first element of tuple is number of rows, second number of columns,
        third number of non-zero values, fourth sprasing [%]

    Examples
    ========
    >>> A = numpy.array([[3, 2, 1], [5, 0, 0], [0, 3, 0]])
    >>> A = scipy.sparse.csr_matrix(A)
    >>> get_info_matrix(A)
        (3, 3, 5, 55.5556)
    '''
    shape = matrix.shape
    nnz = matrix.nnz
    sparsing = round(((nnz+0.0) / ((shape[0] * shape[1]))) * 100, 4)
    return (shape[0], shape[1], nnz, sparsing)

def get_info_vector(vector):
    '''
    Return basic information about vector.

    Parameters
    ==========
    vector : list, numpy array, other...
        vector

    Returns
    =======
    information about vector : tuple
        first element of tuple is length, second number of non-zero values,
        third sprasing [%]

    Examples
    ========
    >>> vec = [3, 2, 1, 0, 0, 0, 2, 3, 0, 0]
    >>> get_info_vector(vec)
        (10, 5, 50.0)
    '''
    length = len(vector)
    nnz = sum(1 for i in vector if not i)
    sparsing = round(((nnz+0.0) / length) * 100, 4)
    return (length, nnz, sparsing)

def result_equals(correct, current, confidence_factor=0.0005):
    '''
    Method compares a list with list correct, reference.

    Values ​​in the list being compared must be within the confidence interval.
    (correct - correct*confidence_factor : correct + correct*confidence_factor)

    If the length of the list correct and current are different additional
    fields should be zero.

    Parameters
    ==========
    correct : list of numbers
        list of reference. list of valid values
    current : list of numbers
        list to compare
    confidence_factor : float
        values ​​in the list being compared must be within the
        confidence interval from (correct - correct*confidence_factor)
        to (correct + correct*confidence_factor)

    Returns
    =======
    errors : list of tuple
        if error append to list tuple (index, different, relative error).

        Different is equal: (correct - current).

        Relative error is equal: (correct - current) / correct.

        If length of the list correct and current are not different and
        additional fields are not zero different and relative error in tuple
        is '#': (index, #, #)

        If correct value is 0 relative error is '#': (index, difference, #)

        If correct value is 0 round(current, 8) should be equal 0.

    Examples
    ========
    >>> l = [1, 2, 3, 4, 0]
    >>> k = [1, 2, 9, 4.1, 0.0000000009]
    >>> result_equals(l, k, 0.1)
        [(2, -6, -2)]

    '''
    result = []
    if len(correct) > len(current):
        end_min = len(current)
        obj_max = correct
    else:
        end_min = len(correct)
        obj_max = current
    for i in range(end_min):
        if correct[i] == 0:
            if round(abs(current[i]), 8) != 0:
                result.append((i, current[i]*(-1), '#'))
        else:
            if (correct[i] > 0 and
                    (current[i] > correct[i]*(1+confidence_factor) or
                    current[i] < correct[i]*(1-confidence_factor))) or \
               (correct[i] < 0 and
                    (current[i] < correct[i]*(1+confidence_factor) or
                    current[i] > correct[i]*(1-confidence_factor))):
                different = correct[i] - current[i]
                result.append((i, different, different/correct[i]))
    for i in range(end_min, len(obj_max)):
        if obj_max[i] != 0:
            return result.append((i, '#', '#'))
    return result

def dict_vector_paths(vector_paths):
    '''
    Method return dict where key is length of vector and value is
    path to vector.

    Parameters
    ==========
    vector_paths : enumerate
        list of paths to vector in numpy format file

    Returns
    =======
    dict vector path : dict
        Diict where key is length of vector and value is
        path to vector.

        dict[len(vector)] = pathVector

    Examples
    ========
    >>> p1 = 'Data/vector_5.npy'
    >>> p2 = 'Data/vector_3.npy'
    >>> dict_vector_paths([p1, p2])
        {3: 'Data/vector_3.npy', 5: 'Data/vector_5.npy'}

    '''
    result = {}
    for path in vector_paths:
        try:
            vector = numpy.load(path)
        except:
            continue
        result[len(vector)] = path
    return result

if __name__ == '__main__':
    pass
